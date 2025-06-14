import argparse
import os
import sys
# from operator import attrgetter
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("/home/lufan/Projects/PensieveV2/LLaVA_NeXT/")
sys.path.append("/home/lufan/Projects/PensieveV2/experiments/")
from tqdm import tqdm

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.utils import disable_torch_init
from llava.conversation import conv_templates, SeparatorStyle

import torch
# import cv2
import numpy as np
import json
from PIL import Image
import requests
import copy
from transformers import set_seed
import warnings
from data.whoops.whoops_utils import get_timestamp
from data.mvbench.load_mvbench import load_data_for_mvbench_test
from decord import VideoReader, cpu
from pathlib import Path

warnings.filterwarnings("ignore")

def get_index(bound, max_frame, num_segments=16, first_idx=0, fps=3):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = [
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ]
    return frame_indices

def load_video(video_path, num_segments=16, bound=None):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0), num_threads=1)
    # total_frame_num = len(vr)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    
    # uniform_sampled_frames = np.linspace(0, max_frame, 
                                        #  max_frames_num, dtype=int)
    frame_idx = get_index(bound, max_frame, num_segments, 
                          first_idx=0, fps=fps) 
    
    # frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames, fps  # (frames, height, width, channels)

def eval_model(args):
    disable_torch_init()
    device = "cuda"
    device_map = "auto"
    # data
    mvbench_data = load_data_for_mvbench_test(args.mvbench_path, args.interested_tasks)
    
    tokenizer, model, image_processor, max_length = load_pretrained_model(args.model_path, 
                                                                          None, 
                                                                          args.model_name, 
                                                                          device_map=device_map, 
                                                                          attn_implementation=None)
    model.eval()
    # output
    answers_file = os.path.expanduser(os.path.join(args.result_path, args.answers_file_name))
    ans_file = open(answers_file, "w")
    print(f"save to {answers_file}")

    for interested_task, task_data in mvbench_data.items():
        # task_data = task_data[83:86]  # TODO
        for line in tqdm(task_data):
            sample_id = line["sample_id"]
            video_path_list = line["video_path_list"]
            question_option = line["question"]
            answer = line["answer"]
            answer_option = line["answer_option"]
            options = line["options"]
            raw_question = line["raw_question"]
            new_task_name = line["new_task_name"]
            read_modality = line["read_modality"]
            use_bound = line["use_bound"]
            bound = line["bound"]
            fps = line["fps"]
            
            # Function to extract frames from video
            if read_modality == 'frame':
                # Load and process video as multi-frames
                if not fps:
                    fps = 3
                frame_indices = get_index(bound, len(video_path_list)-1,
                                          num_segments=args.num_segments, 
                                          first_idx=1, fps=fps) # frame_idx starts from 1
                frame_indices = sorted(list(set(frame_indices)))
                # print(frame_indices)
                # print(len(video_path_list))
                video_path_list_selected = [video_path_list[ii] for ii in frame_indices]
                image_pils = [Image.open(fp).convert("RGB") \
                    for fp in video_path_list_selected]
                image_sizes = [q.size for q in image_pils]
                video_frames = process_images(image_pils, image_processor, model.config)
                # torch.Size([num_imgs, 5, 3, 384, 384]) for anyres
                frames = [_image[:1, ...].to(dtype=torch.float16, device=device) for _image in video_frames]
                assert len(frames) <= args.num_segments
                image_tensors = []
                image_tensors.append(torch.cat(frames, dim=0))
                # image_sizes = [frame.size for frame in image_pils]
                
            elif read_modality == 'video':
                # Load and process video
                assert len(video_path_list) == 1
                video_path_list_selected = video_path_list
                video_frames, fps = load_video(video_path_list[0], args.num_segments, bound)
                # print(video_frames.shape) # (16, 1024, 576, 3)  # (16, 480, 852, 3)
                image_tensors = []
                frames = image_processor.preprocess(
                    video_frames, return_tensors="pt")["pixel_values"].half().cuda()
                # print(frames.shape)  # torch.Size([16, 3, 384, 384])
                image_tensors.append(frames)
                # image feature shape torch.Size([16, 729, 3584]) pooled to torch.Size([16, 196, 3584])
                image_sizes = [frame.size for frame in video_frames]
            else:
                raise NotImplementedError
            
            # Prepare conversation input
            conv_template = "qwen_1_5"
            qs = f"{DEFAULT_IMAGE_TOKEN}\n" + question_option
            assert qs.count(DEFAULT_IMAGE_TOKEN) == len(image_tensors)

            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt_question, 
                                            tokenizer, 
                                            IMAGE_TOKEN_INDEX, 
                                            return_tensors="pt"
                                            ).unsqueeze(0).to(device)
    
            assert "anyres" in args.image_aspect_ratio
            
            with torch.inference_mode():
                # Generate response
                cont = model.generate(
                    input_ids,
                    images=image_tensors,
                    image_sizes=image_sizes,
                    do_sample=args.do_sample,
                    num_beams=args.num_beams,
                    temperature=args.temperature,
                    max_new_tokens=16,
                    modalities=["video"],
                )
                
            if args.mm_vision_select_feature == 'patch':
                if type(image_tensors) is list or image_tensors.ndim == 5:
                    if type(image_tensors) is list:
                        image_tensor_ = [x.unsqueeze(0) if x.ndim == 3 else x for x in image_tensors]
                    
                    images_list = []
                    for image in image_tensor_:
                        if image.ndim == 4:
                            images_list.append(image)
                        else:
                            images_list.append(image.unsqueeze(0))
                    concat_images = torch.cat([image for image in images_list], dim=0)
                    image_tensor_ = concat_images
                    # split_sizes = [image.shape[0] for image in images_list]
                
                image_features = model.encode_images(image_tensor_)
                image_features_pooled = model.get_2dPool(image_features)
                # print(image_features.shape)
                num_vis_tokens = 0
                for image_feature in image_features_pooled:
                    num_vis_tokens += image_feature.shape[-2]
                # num_vis_tokens = image_features.shape[0] * image_features.shape[1]
                # num_vis_tokens = 576
                
            elif args.mm_vision_select_feature in ['video_random_patch', 
                                                   'video_high_info_patch',
                                                   'video_high_info_patch_double_smx']:
                num_vis_tokens = model.model.vision_tower.num_visual_tokens_buffer
                
            else:
                raise NotImplementedError
            
            text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
            outputs = text_outputs.strip()
            # write to disk
            ans_file.write(json.dumps({"sample_id": sample_id,
                                    "image_names": video_path_list_selected,
                                    "question": question_option,
                                    "answer_pred": outputs,
                                    "answer_gt_option":answer_option,
                                    "answer_gt": answer,
                                    "options":options,
                                    "raw_question":raw_question,
                                    "task_name":interested_task,
                                    "new_task_name":new_task_name,
                                    "model_id": args.model_name,
                                    "use_bound":use_bound,
                                    "bound":bound,
                                    "fps":fps,
                                    "read_modality":read_modality,
                                    # "caption_tokens": outputs_per_token_list,
                                    "OtherMetaInfo": {
                                            "num_segments":args.num_segments,
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
    parser.add_argument("--mvbench_path", type=str, default="")
    parser.add_argument("--interested_tasks", type=list, default=[])
    parser.add_argument("--num_segments", type=int, default=16)
    
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--answers_file_name", type=str, default=None)
    
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    
    parser.add_argument("--image_content", type=str, default='', 
                        choices=['intact', 'noised', 'neighbor'])
    parser.add_argument("--decode_method", type=str, default='', 
                        choices=['greedy', 'sample', 'beamsearch'])
    
    args = parser.parse_args()
    
    args.model_path = '/DATA2/yangdingchen/checkpoint/llava-onevision-qwen2-7b-ov'
    args.model_name = 'llava_qwen'
    args.mvbench_path = '/DATA2/yangdingchen/mvbench/'
    args.result_path = args.mvbench_path + 'results/' + get_timestamp() 
    Path(args.result_path).mkdir(parents=True, exist_ok=True)
    
    args.seed = 1234  # TODO
    args.image_content = 'intact'
    args.decode_method = 'greedy'  # TODO
    args.num_segments = 16  # TODO select num of frames uniformally in space
    
    args.interested_tasks = [
        "object_interaction",
        "action_sequence",
        "action_prediction",
        "action_localization",
        "moving_count",
        "fine_grained_pose",
        "character_order",
        "object_shuffle",
        "egocentric_navigation",
        "moving_direction",
        "episodic_reasoning",
        "fine_grained_action",
        "scene_transition",
        "state_change",
        "moving_attribute",
        "action_antonym",
        "unexpected_action",
        "counterfactual_inference",
        "object_existence",
        "action_count",
    ]
    
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
    args.mm_patch_merge_type = this_config["mm_patch_merge_type"]
    args.toxic_codebook_path = this_config["toxic_codebook_path"]
    if args.mm_vision_select_feature in ['low_sim_patch', 'high_sim_patch', 'fix_sim_patch']:
        raise NotImplementedError
        args.mm_vision_reduction_scale = this_config["mm_vision_reduction_scale"]
    elif args.mm_vision_select_feature in ['video_high_info_patch', 'video_high_info_patch_double_smx']:
        args.mm_vision_reduction_scale = this_config["toxic_codebook_thres"]
        assert args.mm_vision_reduction_scale <= 1.0, "with toxic codebook, please assign mm_vision_reduction_scale < 1"
        args.mm_vision_reduction_scale = round(100*args.mm_vision_reduction_scale)
    elif args.mm_vision_select_feature in ['video_random_patch']:
        args.mm_vision_reduction_scale = this_config["random_patch_num"] / 196
        assert args.mm_vision_reduction_scale <= 1.0, "with random please assign random_patch_num < 196"
        args.mm_vision_reduction_scale = round(100*args.mm_vision_reduction_scale)
    else:
        args.mm_vision_reduction_scale = 0
    
    answer_file_prefix = 'llavaOV_mvbench_test_zeroshot_multichoice_video'
    args.answers_file_name = answer_file_prefix + f'_{args.image_content}_{args.decode_method}_{decode_assist}-{args.mm_vision_select_feature}-{args.mm_vision_reduction_scale}.jsonl'
    # args.answers_file_name = answer_file_prefix + f'_{args.image_content}_{args.decode_method}_{decode_assist}.json'
    
    set_seed(args.seed)
    eval_model(args)