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

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.utils import disable_torch_init
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
from pathlib import Path
from transformers import set_seed
from data.whoops.whoops_utils import get_timestamp
from data.muir_bench.muir_utils import load_data_for_muir_test
import warnings

warnings.filterwarnings("ignore")

def eval_model(args):
    disable_torch_init()
    device = "cuda"
    device_map = "auto"
    muir_data = load_data_for_muir_test(args.muir_path)
    
    tokenizer, model, image_processor, max_length = load_pretrained_model(args.model_path, 
                                                                          None, 
                                                                          args.model_name, 
                                                                          device_map=device_map, 
                                                                          attn_implementation=None)  
    # Add any other thing you want to pass in llava_model_args
    model.eval()
    all_data = []
    for domain in args.muir_split:
        print(f"loading {domain} data")
        all_data += muir_data[domain]
        
    answers_file = os.path.expanduser(os.path.join(args.result_path, args.answers_file_name))
    ans_file = open(answers_file, "w")
    print(f"save to {answers_file}")
    
    if args.use_rancd or args.save_logits:
        raise NotImplementedError
        q_nn_file_name = args.q_nn_file_path + \
            f'retrieved_{args.database}_imgs_clip_vit_l14_dino_vit_l14_32nns_nocaps_images_trainval.json'  # TODO
        q_nn_file = json.load(open(q_nn_file_name, 'r'))
        
    if args.save_logits:
        raise NotImplementedError
        if args.logits_file_name is None:
            args.logits_file_name = args.answers_file_name.replace('.json', '.hdf5')
        h5py_file = os.path.expanduser(os.path.join(args.result_path, args.logits_file_name))
        logits_file = h5py.File(h5py_file, 'w')
        
    for line in tqdm(all_data):  # bs=1 only
        sample_id = line["sample_id"]
        image_names = line["image_names"]
        # if len(image_names) > 6:
            # print(f"{sample_id} ha/s more than 6 image, skipped for now")  # TODO
            # continue
        question = line["question"]
        options = line["options"]
        answer = line["answer"]
        raw_question = line["raw_question"]
        task_name = line["task_name"]
        image_type = line["image_type"]
        image_relation = line["image_relation"]
        counterpart_idx = line["counterpart_idx"]
        
        if question.count('<image>') < len(image_names):
            # raise ValueError  # <image> can occur in question option
            print(f"{sample_id} has invalid question with img placeholder, skipped")
            continue
            # question = "<image>"*(len(image_names) - question.count('<image>')) + question

        question = question.replace("<image>", f"{DEFAULT_IMAGE_TOKEN}\n")
        cur_prompt = question
        image_pils = [Image.open(q) for q in image_names]
        image_sizes = [q.size for q in image_pils]
        image_tensor = process_images(image_pils, image_processor, model.config)
        # torch.Size([num_imgs, 5, 3, 384, 384]) for anyres
        # image_tensor = image_tensor[:, :1, ...]  
        # # TODO retain the base image only
        image_tensor = [_image[:1, ...].to(dtype=torch.float16, device=device) for _image in image_tensor]
        # image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
        
        assert question.count(DEFAULT_IMAGE_TOKEN) == len(image_tensor)
        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, 
                                  return_tensors="pt").unsqueeze(0).to(device)
        
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
            split_sizes = [image.shape[0] for image in images_list]
        
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
                max_new_tokens=16,  # copied from /lmms_eval/tasks/muirbench/muirbench.yaml
            )
        
        # decode zeroshot token with intact image input
        outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        # write to disk
        ans_file.write(json.dumps({"sample_id": sample_id,
                                   "image_names": image_names,
                                   "question": cur_prompt,
                                   "answer_pred": outputs,
                                   "answer": answer,
                                   "options":options,
                                   "raw_question":raw_question,
                                   "task_name":task_name,
                                    "image_type":image_type,
                                    "image_relation":image_relation,
                                    "counterpart_idx":counterpart_idx,
                                   "model_id": args.model_name,
                                   # "caption_tokens": outputs_per_token_list,
                                   "OtherMetaInfo": {
                                        "mm_vision_select_feature":args.mm_vision_select_feature,
                                        "mm_vision_reduction_scale":args.mm_vision_reduction_scale,
                                        "image_aspect_ratio":args.image_aspect_ratio,
                                        "mm_patch_merge_type":args.mm_patch_merge_type,
                                        "num_vis_tokens":num_vis_tokens,
                                        "decode":args.decode_method,
                                        "toxic_codebook_path":args.toxic_codebook_path,
                                        "seed":args.seed,
                                       }}) + "\n")
        ans_file.flush()
    
    ans_file.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--muir_path", type=str, default="")
    parser.add_argument("--muir_ques_temp_path", type=str, default="")
    parser.add_argument("--muir_split", type=list, default=[])
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
    
    args = parser.parse_args()
    
    args.model_path = '/DATA2/yangdingchen/checkpoint/llava-onevision-qwen2-7b-ov'
    args.model_name = 'llava_qwen'
    args.muir_path = '/DATA2/yangdingchen/MUIRBENCH/'
    args.muir_split = ["test"]
    args.result_path = args.muir_path + 'results/' + get_timestamp() 
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
        assert not args.do_sample
        args.noise_steps = [999]
        args.kNN = 4
        args.nn_start = 0
    
    decode_assist = 'wo-cd'
    if args.use_rancd:
        raise NotImplementedError
        assert args.image_content == 'intact'
        assert args.decode_method in ['greedy', 'sample', 'beamserch']
        args.oracle_noise_step = 900 # 500
        args.racd_topk = 50
        args.kNN = 2
        decode_assist = 'w-rancd'
        
        args.alpha_noise = 0.01
        args.alpha_nns = 0.02
        args.alpha_base = 1.0
        
        args.nn_start = 0
        args.jsd_thres = None
        
    # args.repetition_penalty = 1.0  # DO NOT support logit postprocessor as llava use -200 image token
        
    this_config = json.load(open(os.path.join(args.model_path, "config.json")))
    args.mm_vision_select_feature = this_config["mm_vision_select_feature"]
    args.image_aspect_ratio = this_config["image_aspect_ratio"]
    args.mm_patch_merge_type = this_config["mm_patch_merge_type"]
    args.toxic_codebook_path = this_config["toxic_codebook_path"]
    if args.mm_vision_select_feature in ['low_sim_patch', 'high_sim_patch', 'fix_sim_patch']:
        args.mm_vision_reduction_scale = this_config["mm_vision_reduction_scale"]
    elif args.mm_vision_select_feature in ['high_info_patch', 'high_info_patch_double_smx', 'high_info_patch_single_smx']:
        args.mm_vision_reduction_scale = this_config["toxic_codebook_thres"]
        assert args.mm_vision_reduction_scale <= 1.0, "with toxic codebook, please assign mm_vision_reduction_scale < 1"
        args.mm_vision_reduction_scale = round(100*args.mm_vision_reduction_scale)
    elif args.mm_vision_select_feature == 'random_patch':
        args.mm_vision_reduction_scale = this_config["random_patch_num"] / 729
        assert args.mm_vision_reduction_scale <= 1.0, "with random please assign random_patch_num < 576"
        args.mm_vision_reduction_scale = round(100*args.mm_vision_reduction_scale)
    else:
        args.mm_vision_reduction_scale = 0
    
    answer_file_prefix = 'llavaOV_muir_test_zeroshot_multichoice_image'
    args.answers_file_name = answer_file_prefix + f'_{args.image_content}_{args.decode_method}_{decode_assist}-{args.mm_vision_select_feature}-{args.mm_vision_reduction_scale}.jsonl'
    # args.answers_file_name = answer_file_prefix + f'_{args.image_content}_{args.decode_method}_{decode_assist}.json'
    
    # args.database = 'coco'
    # args.database_path = f'/DATA3/yangdingchen/{args.database}/images/'
    # args.q_nn_file_path = '/home/lufan/Projects/VCD/experiments/rag/q_nn_files/'
    
    set_seed(args.seed)
    eval_model(args)