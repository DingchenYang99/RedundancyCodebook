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

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math
from pathlib import Path
import copy
import h5py
# import kornia
from transformers import set_seed
from data.whoops.whoops_utils import get_timestamp
from data.seed_bench.seed_img_test_utils import load_data_for_seed_img_test
import warnings

def eval_model(args):
    # Load Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    # data
    seed_img_data = load_data_for_seed_img_test(args.seed_path)
    all_seed_img_data = []
    for domain in args.seed_split:
        print(f"loading {domain} data")
        all_seed_img_data += seed_img_data[domain]
    # seed_img_data = seed_img_data[:2]  # TODO debug
    answers_file = os.path.expanduser(os.path.join(args.result_path, args.answers_file_name))
    ans_file = open(answers_file, "w")
    print(f"save to {answers_file}")
        
    for line in tqdm(all_seed_img_data):  # bs=1 only
        idx = line['sample_id']
        image_path = line['image_name']
        qs = line["question"]
        raw_question = line["raw_question"]
        choice_a = line["choice_a"]
        choice_b = line["choice_b"]
        choice_c = line["choice_c"]
        choice_d = line["choice_d"]
        answer = line["answer"]
        question_id = line["question_id"]
        data_id = line["data_id"]
        question_type_id = line["question_type_id"]
        
        data_type = line["data_type"]
        if data_type not in args.seed_split:
            continue
        
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        # conv.append_message(conv.roles[0], qs + " Please answer this question with one word.")
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        image = Image.open(image_path)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        # get num_vis_tokens
        image_features = model.encode_images(image_tensor.unsqueeze(0).half().cuda())
        num_vis_tokens = image_features.shape[-2]
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        with torch.inference_mode():
            model_output = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=args.do_sample,
                temperature=args.temperature,
                # top_p=args.top_p,
                # top_k=args.top_k,
                num_beams=args.num_beams,
                max_new_tokens=16,
                return_dict_in_generate=True,
                output_scores=False,
                output_hidden_states=False,
                use_cache=True
                )
            output_ids = model_output.sequences
        
        # decode zeroshot token with intact image input
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            
        # probe scores change by replace image tokens
        caption_ids = output_ids[:, input_token_len:-1].clone()  # remove '</s>'
        output_caption_len = caption_ids.shape[1]
        # assert output_caption_len == intact_scores.shape[0]
        # decode per token
        # outputs_per_token_list = [tokenizer.convert_ids_to_tokens(
            # caption_ids[:, i:i+1])[0] for i in range(output_caption_len)]
        
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        
        # write to disk
        ans_file.write(json.dumps({"image_id": idx,
                                   "image_path": image_path,
                                   "question": cur_prompt,
                                   "pred": outputs,
                                   "answer":answer,
                                   "raw_question":raw_question,
                                   "choice_a":choice_a,
                                   "choice_b":choice_b,
                                   "choice_c":choice_c,
                                   "choice_d":choice_d,
                                   "data_type":data_type,
                                   "data_id":data_id,
                                   "question_id":question_id,
                                   "question_type_id":question_type_id,
                                   "model_id": args.model_name,
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
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--seed_path", type=str, default="")
    parser.add_argument("--seed_split", type=list, default=[])
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--answers_file_name", type=str, default=None)
    parser.add_argument("--save_logits", type=bool, default=False)
    parser.add_argument("--logits_file_name", type=str, default=None)
    parser.add_argument("--conv_mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    
    parser.add_argument("--use_rancd", action='store_true', default=False)
    parser.add_argument("--image_content", type=str, default='', 
                        choices=['intact', 'noised', 'neighbor'])
    parser.add_argument("--decode_method", type=str, default='', 
                        choices=['greedy', 'sample', 'beamsearch'])
    
    args = parser.parse_args()
    
    # args.model_path = '/apdcephfs_gy4_303464260/share_303464260/dichyang/opensource_models/llava-v1.5-7b'
    args.model_path = '/apdcephfs_gy4_303464260/share_303464260/dichyang/opensource_models/llava-v1.5-13b'  # TODO
    args.seed_path = '/apdcephfs_cq8/share_1367250/dichyang/data/seed_bench/SEED/'
    args.seed_split = ["image"]  # video
    args.result_path = args.seed_path + 'results/' + get_timestamp() 
    Path(args.result_path).mkdir(parents=True, exist_ok=True)
    
    args.seed = 42   # TODO 42 0 1234
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
    
    decode_assist = 'wo-cd'
    
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
    
    answer_file_prefix = 'llava15_seed-img_zeroshot_multichoice_image'
    args.answers_file_name = answer_file_prefix + f'_{args.image_content}_{args.decode_method}_{decode_assist}-{args.mm_vision_select_feature}-{args.mm_vision_reduction_scale}.json'
    # args.answers_file_name = answer_file_prefix + f'_{args.image_content}_{args.decode_method}_{decode_assist}.json'
    
    # args.database = 'coco'
    # args.database_path = f'/DATA3/yangdingchen/{args.database}/images/'
    # args.q_nn_file_path = '/home/lufan/Projects/VCD/experiments/rag/q_nn_files/'
    
    set_seed(args.seed)
    eval_model(args)
