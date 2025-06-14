import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
import h5py
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
# import kornia
from transformers import set_seed
from pathlib import Path
from data.whoops.whoops_utils import get_timestamp
from data.mme.mme_utils import load_data_mme
     
def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    for interested_task_name in args.interested_task_names:
        answers_file_name = args.result_path + f"/llava15_mme_{interested_task_name}_answers_{args.decode_assist}_{args.mm_vision_select_feature}-{args.mm_vision_reduction_scale}.txt"
        questions = load_data_mme(args.image_folder, interested_task_name)  # [:1]  # debug
        answers_file = os.path.expanduser(answers_file_name)
        # os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        ans_file =  open(answers_file, "w")
        answers_file_json_name = answers_file_name.replace('.txt', '.jsonl')
        answers_file_json = os.path.expanduser(answers_file_json_name)
        ans_file_json =  open(answers_file_json, "w")
            
        if args.save_logits:
            logits_file_name = answers_file_name.replace('.txt', '.hdf5')
            h5py_file = os.path.expanduser(logits_file_name)
            logits_file = h5py.File(h5py_file, 'w')
        
        for idx, line in enumerate(tqdm(questions)):
            # idx = line["question_id"]
            image_file = line["image"]
            # print(image_file)
            qs = line["text"]
            gt_ans = line["label"]
            
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

            image = Image.open(image_file)
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            # get num_vis_tokens
            image_features = model.encode_images(image_tensor.unsqueeze(0).half().cuda())
            num_vis_tokens = image_features.shape[-2]
            
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            # get base logits
            with torch.inference_mode():
                model_output = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    # top_k=args.top_k,
                    num_beams=args.num_beams,
                    max_new_tokens=16,
                    return_dict_in_generate=True,
                    output_scores=False,
                    output_hidden_states=False,
                    use_cache=True
                    )
                output_ids = model_output.sequences
                
            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                
            # probe scores change by replace image tokens
            caption_ids = output_ids[:, input_token_len:-1].clone()  # remove '</s>'
            # print(caption_ids)
            output_caption_len = caption_ids.shape[1]
            # assert output_caption_len == intact_scores.shape[0]
            # print(input_token_len, output_caption_len)
            # decode per token
            # outputs_per_token_list = [tokenizer.convert_ids_to_tokens(
                # caption_ids[:, i:i+1])[0] for i in range(output_caption_len)]
            # outputs_per_token_list = [tokenizer.batch_decode(
            #     caption_ids[:, i:i+1], skip_special_tokens=True)[0] for i in range(output_caption_len)]
            
            
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            ans_file.write(image_file.split("/")[-1] + "\t" + cur_prompt + "\t" + gt_ans + "\t" + outputs + "\n")
            # print(gt_ans, outputs)
            ans_file_json.write(json.dumps({"question_id": idx,
                                        "prompt": cur_prompt,
                                        "text": outputs,
                                        "label": gt_ans,
                                        "model_id": model_name,
                                        "image": image_file,  # .split("/")[-1],
                                        # "caption_tokens": outputs_per_token_list,
                                   "OtherMetaInfo": {
                                        "mm_vision_select_feature":args.mm_vision_select_feature,
                                        "mm_vision_reduction_scale":args.mm_vision_reduction_scale,
                                        "image_aspect_ratio":args.image_aspect_ratio,
                                        # "mm_patch_merge_type":args.mm_patch_merge_type,
                                        "num_vis_tokens":num_vis_tokens,
                                        "decode":args.decode_method,
                                        "toxic_codebook_path":args.toxic_codebook_path,
                                        }}) + "\n")
            ans_file_json.flush()
            ans_file.flush()
                
        ans_file.close()
        ans_file_json.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--question_file", type=str, default="")
    parser.add_argument("--answers_file", type=str, default="")
    parser.add_argument("--save_logits", type=bool, default=False)
    parser.add_argument("--logits_file_name", type=str, default=None)
    parser.add_argument("--image_content", type=str, default='intact', 
                        choices=['intact', 'noised', 'neighbor', 'ablate'])
    
    parser.add_argument("--conv_mode", type=str, default="llava_v1")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    
    args.mme_path = "/apdcephfs_cq8/share_1367250/dichyang/data/benchmarks/mme/"
    # args.model_path = '/apdcephfs_gy4_303464260/share_303464260/dichyang/opensource_models/llava-v1.5-7b'
    args.model_path = '/apdcephfs_gy4_303464260/share_303464260/dichyang/opensource_models/llava-v1.5-13b'  # TODO
    
    args.result_path = args.mme_path + 'results/' + get_timestamp()
    args.image_folder = args.mme_path + 'MME_Benchmark_release_version/'
    Path(args.result_path).mkdir(parents=True, exist_ok=True)
    print(f"save to {args.result_path}")
    
    args.seed = 42   # TODO 42 0 1234
    args.image_content = 'intact'
    args.decode_method = 'greedy'  # TODO
    
    args.interested_task_names = [
        "existence", 
        "count", 
        "position",
        "color",
        "posters", 
        "celebrity", 
        "scene", 
        "landmark", 
        "artwork", 
        "OCR",
        ]  # TODO
    
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
    
    args.decode_assist = 'wo-cd'
    
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
    
    set_seed(args.seed)
    eval_model(args)
