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

from data.mme.mme_utils import load_data_mme
from data.whoops.whoops_utils import get_timestamp
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
    for interested_task_name in args.interested_task_names:
        answers_file_name = args.result_path + f"/llavaNext_mme_{interested_task_name}_answers_{args.decode_assist}_{args.mm_vision_select_feature}-{args.mm_vision_reduction_scale}.txt"
        questions = load_data_mme(args.image_folder, interested_task_name)  # [:2]  # TODO debug
        answers_file = os.path.expanduser(answers_file_name)
        # os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        print(f"save to {answers_file}")
        ans_file = open(answers_file, "w")
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
            
            image = Image.open(image_file)
            image_sizes = [image.size]
            image_tensor = process_images([image], image_processor, model.config)
            image_tensor = image_tensor[:, :1, ...]  # TODO retain base image only, [1,5,3,336,336]
            # print(image_tensor.shape)
            if len(image_tensor.shape) == 5:
                len_sub_imgs = image_tensor.shape[1] 
            elif len(image_tensor.shape) == 4:
                raise ValueError
                len_sub_imgs = image_tensor.shape[0]
                assert args.image_aspect_ratio == "naive"
            else:
                raise ValueError
            image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
            
            # len_sub_imgs = len(image_tensor)
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
            # print(len(image_features))
            for image_feature in image_features:
                # print(image_feature.shape)
                num_vis_tokens += image_feature.shape[-2]
            # num_vis_tokens = image_features.shape[0] * image_features.shape[1]
            # num_vis_tokens = 576
            
            # get model response
            with torch.inference_mode():
                cont = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=args.do_sample,
                    num_beams=args.num_beams,
                    temperature=args.temperature,
                    max_new_tokens=16,  # follow lmms-eval setting
                    )
            
            if args.save_bin_retain_mask:
                high_info_index_bin = model.model.vision_tower.patch_retain_mask_bin
                assert high_info_index_bin.ndim == 2 and high_info_index_bin.shape[0] == 1
                high_info_index_bin_list = high_info_index_bin[0].tolist()
            else:
                high_info_index_bin_list = None
                
            # decode zeroshot token with intact image input
            outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
            outputs = outputs.strip()
            
            # write to disk
            ans_file.write(image_file.split("/")[-1] + "\t" + cur_prompt + "\t" + gt_ans + "\t" + outputs + "\n")
            ans_file_json.write(json.dumps({"question_id": idx,
                                        "prompt": cur_prompt,
                                        "text": outputs,
                                        "label": gt_ans,
                                        "model_id": args.model_name,
                                        "image": image_file,  # .split("/")[-1],
                                        # "caption_tokens": outputs_per_token_list,
                                   "OtherMetaInfo": {
                                        "mm_vision_select_feature":args.mm_vision_select_feature,
                                        "mm_vision_reduction_scale":args.mm_vision_reduction_scale,
                                        "image_aspect_ratio":args.image_aspect_ratio,
                                        "mm_patch_merge_type":args.mm_patch_merge_type,
                                        "num_vis_tokens":num_vis_tokens,
                                        "decode":args.decode_method,
                                        "toxic_codebook_path":args.toxic_codebook_path,
                                        "high_info_index_bin_list":high_info_index_bin_list,
                                       }}) + "\n")
            ans_file_json.flush()
            ans_file.flush()
            
            if args.save_logits:
                raise NotImplementedError
        ans_file.close()
        ans_file_json.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--mme_path", type=str, default="")
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--answers_file", type=str, default="")
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
    parser.add_argument("--racd_topk", type=int)
    args = parser.parse_args()
    
    args.mme_path = "/DATA3/yangdingchen/mme/"
    args.model_path = '/DATA3/yangdingchen/checkpoint/llama3-llava-next-8b'
    args.model_name = 'llava_llama3'
    args.result_path = args.mme_path + 'results/' + get_timestamp()
    args.image_folder = args.mme_path + 'MME_Benchmark_release_version/'
    Path(args.result_path).mkdir(parents=True, exist_ok=True)
    print(f"save to {args.result_path}")
    
    args.seed = 42   # TODO 42 0 1234
    args.image_content = 'intact'
    args.decode_method = 'greedy'  # TODO
    args.use_rancd = False
    args.save_logits = False
    args.save_bin_retain_mask = False  # TODO
    
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
    
    if args.save_logits:
        raise NotImplementedError
    
    args.decode_assist = 'wo-cd'
    if args.use_rancd:
        raise NotImplementedError
    
    this_config = json.load(open(os.path.join(args.model_path, "config.json")))
    args.mm_vision_select_feature = this_config["mm_vision_select_feature"]
    args.image_aspect_ratio = this_config["image_aspect_ratio"]
    args.mm_patch_merge_type = this_config["mm_patch_merge_type"]
    args.toxic_codebook_path = this_config["toxic_codebook_path"]
    if args.mm_vision_select_feature == 'low_sim_patch' or args.mm_vision_select_feature == 'high_sim_patch' or args.mm_vision_select_feature == 'fix_sim_patch':
        args.mm_vision_reduction_scale = this_config["mm_vision_reduction_scale"]
    elif args.mm_vision_select_feature in ['high_info_patch', 'high_info_patch_flatten']:
        args.mm_vision_reduction_scale = this_config["toxic_codebook_thres"]
        # print(args.mm_vision_reduction_scale)
        assert args.mm_vision_reduction_scale <= 1.0, "with toxic codebook, please assign mm_vision_reduction_scale < 1"
        args.mm_vision_reduction_scale = round(100*args.mm_vision_reduction_scale)
        # print(args.mm_vision_reduction_scale)
    elif args.mm_vision_select_feature == 'random_patch':
        args.mm_vision_reduction_scale = this_config["random_patch_num"] / 576
        assert args.mm_vision_reduction_scale <= 1.0, "with random please assign random_patch_num < 576"
        args.mm_vision_reduction_scale = round(100*args.mm_vision_reduction_scale)
    else:
        args.mm_vision_reduction_scale = 0
    
    set_seed(args.seed)
    eval_model(args)