import argparse
import torch
import json
from tqdm import tqdm
import shortuuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("/home/lufan/Projects/PensieveV2/LLaVA_NeXT/")

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token

from PIL import Image
import math
from pathlib import Path
import copy
import h5py
# import kornia
from transformers import set_seed
from data.whoops.whoops_utils import get_timestamp
from data.flickr30k_test.flickr30k_test_utils import *
import warnings

warnings.filterwarnings("ignore")

def eval_model(args):
    # Load Model
    disable_torch_init()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_map = "auto"
    # Add any other thing you want to pass in llava_model_args
    tokenizer, model, image_processor, max_length = load_pretrained_model(args.model_path, 
                                                                          None, 
                                                                          args.model_name, 
                                                                          device_map=device_map, 
                                                                          attn_implementation=None) 
    model.eval()
    model.tie_weights()  # TODO
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
        # qs = "Provide a one-sentence caption for the provided image."  # copy from llava1.5 paper and EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/tasks/flickr30k/utils.py
        # qs = "Provide a description for the image."
        # qs = "Please describe this image. Write 2 to 3 sentences."
        qs = "Describe the provided image in one sentence.\nPlease provide detailed and specific descriptions of the key information in the picture."
        # qs = "Please describe this image in detail."  # dense caption # TODO
        cur_prompt = qs
        
        image_path = image_file
        image = Image.open(image_path)
        image_sizes = [image.size]
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = image_tensor[:, :1, ...]  # TODO
        # print(image_tensor.shape)
        if len(image_tensor.shape) == 5:
            len_sub_imgs = image_tensor.shape[1] 
        elif len(image_tensor.shape) == 4:
            raise ValueError
            len_sub_imgs = image_tensor.shape[0]
            assert args.image_aspect_ratio == "naive"
        else:
            raise ValueError
        # print(image_tensor.shape)
        image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
        
        # question = ""
        assert len_sub_imgs == 1, 'support base image only for now 241030'  # TODO
        # for _ in range(len_sub_imgs):
            # question += DEFAULT_IMAGE_TOKEN + '\n'
        # question += qs
        question = DEFAULT_IMAGE_TOKEN + '\n' + qs
        # print(question)
        
        conv_template = "llava_llama_3" # Make sure you use correct chat template for different models
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, 
                                          return_tensors="pt").unsqueeze(0).to(device)
        
        # get num_visual_tokens
        # if args.image_aspect_ratio == "naive":
            # image_tensor = torch.stack(image_tensor, dim=0)
            # image_tensor_ = image_tensor.detach().clone()
        assert "anyres" in args.image_aspect_ratio
        if type(image_tensor) is list or image_tensor.ndim == 5:
            if type(image_tensor) is list:
                image_tensor_ = [x.unsqueeze(0) if x.ndim == 3 else x for x in image_tensor]
            
            images_list = []
            for image in image_tensor_:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))
            concat_images = torch.cat([image for image in images_list], dim=0)
            image_tensor_ = concat_images
            
        image_features = model.encode_images(image_tensor_)
        # print(image_features.shape)
        num_vis_tokens = 0
        for image_feature in image_features:
            num_vis_tokens += image_feature.shape[-2]
        # num_vis_tokens = image_features.shape[0] * image_features.shape[1]
        # num_vis_tokens = 576
        
        with torch.inference_mode():
            cont = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=args.do_sample,
                num_beams=args.num_beams,
                temperature=args.temperature,
                max_new_tokens=256,  # 64 is copied from lmms_eval/tasks/flickr30k/flickr30k_test_lite.yaml, 256 is the same as nocaps
                )
            # tuple, (num_new_tokens,) [1, num_vocab=32000]
            # scores_tuple = model_output.scores
            # intact_scores = torch.cat(scores_tuple[:-2], dim=0)  # remove '.' and '</s>'
            # print(output_scores[:3, :20])
            # output_ids = model_output.sequences
            
        # input_token_len = input_ids.shape[1]
        # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        # if n_diff_input_output > 0:
            # print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        
        # # probe scores change by replace image tokens
        # caption_ids = output_ids[:, input_token_len:-2].clone()  # remove '.' and '</s>'
        # output_caption_len = caption_ids.shape[1]
        # # decode per token
        # outputs_per_token_list = [tokenizer.convert_ids_to_tokens(
        #     caption_ids[:, i:i+1])[0] for i in range(output_caption_len)]
        # # outputs_per_token_list = [tokenizer.batch_decode(
        # #     caption_ids[:, i:i+1], skip_special_tokens=True)[0] for i in range(output_caption_len)]
        # if args.num_beams == 1:
        #     assert output_caption_len == intact_scores.shape[0]
        # assert len(outputs_per_token_list) == output_caption_len

        # decode zeroshot token with intact image input
        outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        # write to disk
        ans_file.write(json.dumps({"image_id": idx,
                                   "question": cur_prompt,
                                   "caption_pred": outputs,
                                   "caption_gt": caption_gt,
                                   "domain": domain,
                                #    "caption_tokens": outputs_per_token_list,
                                   "model_id": args.model_name,
                                   "image": image_file,
                                   "OtherMetaInfo": {
                                        "mm_vision_select_feature":args.mm_vision_select_feature,
                                        "mm_vision_reduction_scale":args.mm_vision_reduction_scale,
                                        "image_aspect_ratio":args.image_aspect_ratio,
                                        "mm_patch_merge_type":args.mm_patch_merge_type,
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
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--flickr_path", type=str, default="")
    parser.add_argument("--flickr_split", type=list, default=[])
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--answers_file_name", type=str, default=None)
    parser.add_argument("--save_logits", type=bool, default=False)
    parser.add_argument("--logits_file_name", type=str, default=None)
    
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    
    parser.add_argument("--use_rancd", action='store_true', default=False)
    parser.add_argument("--image_content", type=str, default='', 
                        choices=['intact', 'noised', 'neighbor'])
    parser.add_argument("--decode_method", type=str, default='', 
                        choices=['greedy', 'sample', 'beamsearch'])
    parser.add_argument("--noise_steps", type=list, help="noise step list")
    parser.add_argument("--oracle_noise_step", type=int, default=500)
    
    parser.add_argument("--kNN", type=int)
    args = parser.parse_args()
    
    args.model_path = '/DATA3/yangdingchen/checkpoint/llama3-llava-next-8b'
    args.model_name = 'llava_llama3'
    args.flickr_path = '/DATA2/yangdingchen/flickr30k/'
    args.flickr_img_path = args.flickr_path + 'images/'
    args.flickr_split = ["test"]
    args.result_path = args.flickr_path + 'results/' + get_timestamp() 
    Path(args.result_path).mkdir(parents=True, exist_ok=True)
    
    args.seed = 42  # TODO
    args.image_content = 'intact'
    args.decode_method = 'greedy'  # TODO
    args.use_rancd = False
    args.save_logits = False
    
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
    
    if args.save_logits:
        raise NotImplementedError
    
    decode_assist = 'wo-cd'
    # args.repetition_penalty = 1.0  # DO NOT support logit postprocessor as llava use -200 image token
        
    this_config = json.load(open(os.path.join(args.model_path, "config.json")))
    args.mm_vision_select_feature = this_config["mm_vision_select_feature"]
    args.image_aspect_ratio = this_config["image_aspect_ratio"]
    args.mm_patch_merge_type = this_config["mm_patch_merge_type"]
    args.toxic_codebook_path = this_config["toxic_codebook_path"]
    if args.mm_vision_select_feature in ['low_sim_patch', 'high_sim_patch', 'fix_sim_patch']:
        args.mm_vision_reduction_scale = this_config["mm_vision_reduction_scale"]
    elif args.mm_vision_select_feature in ['high_info_patch', 'high_info_patch_flatten']:
        args.mm_vision_reduction_scale = this_config["toxic_codebook_thres"]
        assert args.mm_vision_reduction_scale <= 1.0, "with toxic codebook, please assign mm_vision_reduction_scale < 1"
        args.mm_vision_reduction_scale = round(100*args.mm_vision_reduction_scale)
    elif args.mm_vision_select_feature == 'random_patch':
        args.mm_vision_reduction_scale = this_config["random_patch_num"] / 576
        assert args.mm_vision_reduction_scale <= 1.0, "with random please assign random_patch_num < 576"
        args.mm_vision_reduction_scale = round(100*args.mm_vision_reduction_scale)
    else:
        args.mm_vision_reduction_scale = 0
    
    answer_file_prefix = 'llavaNext_flickr30k_test_zeroshot_captions_image'
    args.answers_file_name = answer_file_prefix + f'_{args.image_content}_{args.decode_method}_{decode_assist}-{args.mm_vision_select_feature}-{args.mm_vision_reduction_scale}.json'
    # args.answers_file_name = answer_file_prefix + f'_{args.image_content}_{args.decode_method}_{decode_assist}.json'
    
    set_seed(args.seed)
    eval_model(args)
