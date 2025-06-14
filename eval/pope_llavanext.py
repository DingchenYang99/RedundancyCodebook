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
    # data
    for question_type in args.question_types:
        question_file = args.pope_path + f"{args.dataset_name}/{args.dataset_name}_pope_{question_type}.json"
        questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]  # [:2]  # TODO
        
        answers_file_name = args.answers_file_name.replace("_pope_", f"_{args.dataset_name}_pope_{question_type}_")
        
        answers_file = os.path.expanduser(os.path.join(args.result_path, answers_file_name))
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
            
        for line in tqdm(questions):
            idx = line["question_id"]
            # print(idx)
            image_file = line["image"]
            if args.dataset_name == "coco" or args.dataset_name == "aokvqa":
                image_file = image_file.split("_")[-1]
                
            qs = line["text"]
            gt_ans = line["label"]
            
            cur_prompt = qs
            
            image_path = os.path.join(args.image_folder, image_file)
            if not os.path.exists(image_path):
                print(f"{idx} has invalid image, skipped")
                continue
            image = Image.open(image_path)
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
            question = DEFAULT_IMAGE_TOKEN + '\n' + qs + "\nAnswer the question using a single word or phrase."  # TODO
            # copied from /lmms_eval/tasks/pope/utils.py
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

            if args.use_rancd:
                raise NotImplementedError
            
            with torch.inference_mode():
                cont = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=args.do_sample,
                    num_beams=args.num_beams,
                    temperature=args.temperature,
                    max_new_tokens=16,
                    )
            # decode zeroshot token with intact image input
            outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
            outputs = outputs.strip()
            # write to disk
            ans_file.write(json.dumps({"question_id": idx,
                                    "image": image_file,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "label": gt_ans,
                                    #    "caption_tokens": outputs_per_token_list,
                                    "model_id": args.model_name,
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
        print(f"{question_type} finished")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--pope_path", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default="")
    # parser.add_argument("--question_file", type=str, default="")
    # parser.add_argument("--answers_file", type=str, default="")
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
    
    args.pope_path = "/home/lufan/Projects/VCD/experiments/data/POPE/"
    args.model_path = '/DATA3/yangdingchen/checkpoint/llama3-llava-next-8b'
    args.model_name = 'llava_llama3'
    args.result_path = '/DATA3/yangdingchen/pope/results/' + get_timestamp()
    Path(args.result_path).mkdir(parents=True, exist_ok=True)
    
    args.seed = 42   # TODO 42 0 1234
    args.image_content = 'intact'
    args.decode_method = 'greedy'  # TODO
    args.use_rancd = False
    args.save_logits = False
    
    args.dataset_name = "coco"  # TODO
    args.question_types = ["random", "popular", "adversarial"]  # TODO
    # args.question_types = ["random"]
    # args.question_types = ["popular"]
    # args.question_types = ["adversarial"]
    
    if args.dataset_name == "coco" or args.dataset_name == "aokvqa":
        args.image_folder = "/DATA3/yangdingchen/coco/images"
    elif args.dataset_name == "gqa":
        args.image_folder = "/DATA3/yangdingchen/gqa/images"
    else:
        raise NotImplementedError
    
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
    
    answer_file_prefix = 'llavaNext_pope_closeEndVqa_answers_image'
    args.answers_file_name = answer_file_prefix + f'_{args.image_content}_{args.decode_method}_{decode_assist}-{args.mm_vision_select_feature}-{args.mm_vision_reduction_scale}.jsonl'
    # args.answers_file_name = answer_file_prefix + f'_{args.image_content}_{args.decode_method}_{decode_assist}.json'
    set_seed(args.seed)
    eval_model(args)