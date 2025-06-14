import argparse
import torch
import json
from tqdm import tqdm
import shortuuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/apdcephfs_cq8/share_1367250/dichyang/code/Projects/VCD/experiments/')
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math
from pathlib import Path
import h5py
# import kornia
from transformers import set_seed
from data.whoops.whoops_utils import get_timestamp
from data.flickr30k_test.flickr30k_test_utils import *


def eval_model(args):
    # Load Model
    disable_torch_init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, device=device)
    
    flickr_data = load_data_for_flickr30k_test(args.flickr_path)
    all_flickr_data = []
    for domain in args.flickr_split:
        print(f"loading {domain} data")
        all_flickr_data += flickr_data[domain]
    # all_flickr_data = all_flickr_data[:2]  # TODO debug
    
    answers_file = os.path.expanduser(os.path.join(args.result_path, args.answers_file_name))
    ans_file = open(answers_file, "w")
    print(f"save to {answers_file}")
    
    for line in tqdm(all_flickr_data):  # bs=1 only
        idx = line['image_id']
        image_file = line['file_name']
        caption_gt = line['gt_caption']
        domain = line['domain']
        # qs = "Provide a one-sentence caption for the provided image."  # copy from llava1.5 paper TODO
        qs = "Describe the provided image in one sentence.\nPlease provide detailed and specific descriptions of the key information in the picture."  
        
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        # conv.append_message(conv.roles[0], qs + instruct)  # TODO
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # [bs=1, num_input_token=69]
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, 
                                          return_tensors='pt').unsqueeze(0).cuda()  
        
        image_path = image_file
        image = Image.open(image_path)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        # print(image_tensor.shape)  # [3, 336, 336]  
        
        image_features = model.encode_images(image_tensor.unsqueeze(0).half().cuda())
        num_vis_tokens = image_features.shape[-2]
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            model_output = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                num_beams=args.num_beams,
                do_sample=args.do_sample,
                # top_p=args.top_p,
                # top_k=args.top_k,
                temperature=args.temperature,
                # repetition_penalty=args.repetition_penalty,
                max_new_tokens=64,
                return_dict_in_generate=True,
                output_scores=False,
                output_hidden_states=False,
                use_cache=True
                )
            # tuple, (num_new_tokens,) [1, num_vocab=32000]
            # scores_tuple = model_output.scores
            # intact_scores = torch.cat(scores_tuple[:-2], dim=0)  # remove '.' and '</s>'
            # print(output_scores[:3, :20])
            output_ids = model_output.sequences
            
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        
        # probe scores change by replace image tokens
        caption_ids = output_ids[:, input_token_len:-2].clone()  # remove '.' and '</s>'
        output_caption_len = caption_ids.shape[1]
        # decode per token
        # outputs_per_token_list = [tokenizer.convert_ids_to_tokens(
            # caption_ids[:, i:i+1])[0] for i in range(output_caption_len)]
        # outputs_per_token_list = [tokenizer.batch_decode(
        #     caption_ids[:, i:i+1], skip_special_tokens=True)[0] for i in range(output_caption_len)]
        # if args.num_beams == 1:
            # assert output_caption_len == intact_scores.shape[0]
        # assert len(outputs_per_token_list) == output_caption_len

        # decode zeroshot token with intact image input
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        # write to disk
        ans_file.write(json.dumps({"image_id": idx,
                                   "question": cur_prompt,
                                   "caption_pred": outputs,
                                   "caption_gt": caption_gt,
                                   "domain": domain,
                                #    "caption_tokens": outputs_per_token_list,
                                   "model_id": model_name,
                                   "image": image_file,
                                   "OtherMetaInfo": {
                                        "mm_vision_select_feature":args.mm_vision_select_feature,
                                        "mm_vision_reduction_scale":args.mm_vision_reduction_scale,
                                        "image_aspect_ratio":args.image_aspect_ratio,
                                        # "mm_patch_merge_type":args.mm_patch_merge_type,
                                        "num_vis_tokens":num_vis_tokens,
                                        "decode":args.decode_method,
                                        "toxic_codebook_path":args.toxic_codebook_path,
                                        }}) + "\n")
        ans_file.flush()
            
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--flickr_path", type=str, default="")
    parser.add_argument("--flickr_split", type=list, default=[])
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--answers_file_name", type=str, default=None)
    parser.add_argument("--save_logits", type=bool, default=False)
    parser.add_argument("--logits_file_name", type=str, default=None)
    
    parser.add_argument("--conv_mode", type=str, default="llava_v1")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    
    parser.add_argument("--image_content", type=str, default='', 
                        choices=['intact', 'noised', 'neighbor'])
    parser.add_argument("--decode_method", type=str, default='', 
                        choices=['greedy', 'sample', 'beamsearch'])
    
    args = parser.parse_args()
    
    # args.model_path = '/apdcephfs_gy4_303464260/share_303464260/dichyang/opensource_models/llava-v1.5-7b'
    args.model_path = '/apdcephfs_gy4_303464260/share_303464260/dichyang/opensource_models/llava-v1.5-13b'  # TODO
    args.flickr_path = '/apdcephfs_cq8/share_1367250/dichyang/data/benchmarks/flickr30k/'
    args.flickr_img_path = args.flickr_path + 'images/'
    args.flickr_split = ["test"]
    args.result_path = args.flickr_path + 'results/' + get_timestamp() 
    Path(args.result_path).mkdir(parents=True, exist_ok=True)
    
    args.seed = 42   # TODO 42 0 1234
    args.image_content = 'intact'
    args.decode_method = 'greedy'  # TODO
    
    if args.decode_method == 'greedy':
        args.num_beams = 1
        args.do_sample = False
    elif args.decode_method == 'sample':
        args.num_beams = 1
        args.do_sample = True
    elif args.decode_method == 'beamsearch':
        args.num_beams = 3
        args.do_sample = False
    else:
        raise NotImplementedError
    
    decode_assist = 'wo-cd'
        
    # args.repetition_penalty = 1.0  # DO NOT support logit postprocessor as llava use -200 image token
        
    this_config = json.load(open(os.path.join(args.model_path, "config.json")))
    args.mm_vision_select_feature = this_config["mm_vision_select_feature"]
    args.image_aspect_ratio = this_config["image_aspect_ratio"]
    # args.mm_patch_merge_type = this_config["mm_patch_merge_type"]
    args.toxic_codebook_path = this_config["toxic_codebook_path"]
    if args.mm_vision_select_feature in ['low_sim_patch', 'high_sim_patch', 'fix_sim_patch']:
        args.mm_vision_reduction_scale = this_config["mm_vision_reduction_scale"]
    elif args.mm_vision_select_feature in ['high_info_patch', 'high_info_patch_flatten', 'high_info_patch_fix_num']:
        args.mm_vision_reduction_scale = this_config["toxic_codebook_thres"]
        assert args.mm_vision_reduction_scale <= 1.0, "with toxic codebook, please assign mm_vision_reduction_scale < 1"
        args.mm_vision_reduction_scale = round(100*args.mm_vision_reduction_scale)
    elif args.mm_vision_select_feature == 'random_patch':
        args.mm_vision_reduction_scale = this_config["random_patch_num"] / 576
        assert args.mm_vision_reduction_scale <= 1.0, "with random please assign random_patch_num < 576"
        args.mm_vision_reduction_scale = round(100*args.mm_vision_reduction_scale)
    else:
        args.mm_vision_reduction_scale = 0
    
    answer_file_prefix = 'llava15_flickr30k_test_zeroshot_captions_image'
    args.answers_file_name = answer_file_prefix + f'_{args.image_content}_{args.decode_method}_{decode_assist}-{args.mm_vision_select_feature}-{args.mm_vision_reduction_scale}.json'
    
    set_seed(args.seed)
    eval_model(args)
