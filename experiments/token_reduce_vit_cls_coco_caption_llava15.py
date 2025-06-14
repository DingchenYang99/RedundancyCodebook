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
from data.nocaps.nocaps_utils import *
from data.cococap.cocoutils import load_data_for_token_reduc_exp

def eval_model(args):
    # Load Model
    disable_torch_init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, device=device)
    model.eval()
    # input_word_embeddings = model.get_model().get_input_embeddings().weight
    # print(input_word_embeddings.shape)  # torch.Size([32000, 4096])
    
    # nocaps_data = load_data_for_nocaps(args.nocaps_path)
    coco_data = load_data_for_token_reduc_exp(args.coco_anno_path)[args.coco_split]
    
    coco_data = coco_data[:500]  # TODO
    # instruct = " please be precise and faithful to the image."
    
    for n_scale in args.n_list:
        args.mm_vision_reduction_scale = n_scale
        model.model.vision_tower.patch_reduction_scale = n_scale
        
        args.result_path = args.result_path_template + get_timestamp()
        Path(args.result_path).mkdir(parents=True, exist_ok=True)
        
        args.answers_file_name = args.answer_file_prefix + f'_{args.image_content}_{args.decode_method}-{args.mm_vision_select_feature}-{args.mm_vision_reduction_scale}.jsonl'
        
        answers_file = os.path.expanduser(os.path.join(args.result_path, args.answers_file_name))
        ans_file = open(answers_file, "w")
        print(f"save to {answers_file}")
        
        if args.save_logits:
            # if args.logits_file_name is None:
            args.logits_file_name = args.answers_file_name.replace('.jsonl', '.hdf5')
            h5py_file = os.path.expanduser(os.path.join(args.result_path, args.logits_file_name))
            logits_file = h5py.File(h5py_file, 'w')
        
        for line in tqdm(coco_data):  # bs=1 only
            idx = line['image_id']
            image_file = line['file_name']
            caption_gt = line['gt_caption']
            # qs = "Provide a one-sentence caption for the provided image."  # copy from llava1.5 paper TODO
            # qs = "This image shows a "  # for per-token exp
            # qs = "What does this image show? Answer with one word."
            qs = "What does this image show? You must provide a single word that best describes the image."
            
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
            
            image_path = os.path.join(args.coco_path, image_file)
            image = Image.open(image_path)
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            # print(image_tensor.shape)  # [3, 336, 336]
            
            image_features = model.encode_images(image_tensor.unsqueeze(0).half().cuda())
            num_vis_tokens = image_features.shape[-2]
            # torch.save(image_features, os.path.join(args.result_path, f"coco_{args.coco_split}_{idx}_{args.mm_vision_select_feature}_{num_vis_tokens}-patches.pt"))
            # print((image_features.shape))
            
            if args.save_visual_features:
                # image_token_insert_places = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist()
                # print(image_token_insert_places)  # [45]  TODO important for print attn scores
                assert image_features.ndim == 3
                torch.save(image_features, os.path.join(
                    args.result_path, f"coco_{args.coco_split}_{idx}_{args.mm_vision_select_feature}_idx-{args.mm_vision_reduction_scale}_numPatches-{num_vis_tokens}.pt"))
                
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            # get base logits
            with torch.inference_mode():
                model_output = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    num_beams=args.num_beams,
                    do_sample=args.do_sample,
                    top_p=args.top_p,
                    # top_k=args.top_k,
                    temperature=args.temperature,
                    # repetition_penalty=args.repetition_penalty,
                    max_new_tokens=8,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_hidden_states=False,
                    output_attentions=args.save_attns,
                    use_cache=True
                    )
                # tuple, (num_new_tokens,) [1, num_vocab=32000]
                scores_tuple = model_output.scores
                # print(len(scores_tuple))
                # intact_scores = torch.cat(scores_tuple[:-2], dim=0)  # remove '.' and '</s>'
                # intact_scores = torch.cat(scores_tuple[:-1], dim=0)  # remove '</s>'
                # print(output_scores[:3, :20])
                output_ids = model_output.sequences
                # hidden_states = model_output.hidden_states
                # print(len(hidden_states))  # 19
                # print(len(hidden_states[0]))  # 33
                if args.save_attns:
                    output_attention_scores = model_output.attentions
                    # print(len(output_attentions))  # 3 == output_ids.shape[-1]
                    assert len(output_attention_scores) == output_ids.shape[-1]
                    # print(len(output_attentions[0]))  # 32
                    
            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            
            # probe scores change by replace image tokens
            # caption_ids = output_ids[:, input_token_len:-2].clone()  # remove '.' and '</s>'
            # caption_ids = output_ids[:, input_token_len:-1].clone()  # remove '</s>' 
            caption_ids = output_ids[:, input_token_len:].clone()  # TODO 
            output_caption_len = caption_ids.shape[1]
            
            # decode per token
            outputs_per_token_list = [tokenizer.convert_ids_to_tokens(
                caption_ids[:, i:i+1])[0] for i in range(output_caption_len)]
            # outputs_per_token_list = [tokenizer.batch_decode(
            #     caption_ids[:, i:i+1], skip_special_tokens=True)[0] for i in range(output_caption_len)]
            if outputs_per_token_list[0] == '\u2581A':
                outputs_per_token_list = outputs_per_token_list[1:]
                caption_ids = caption_ids[:, 1:]
                scores_tuple = scores_tuple[1:]
                output_caption_len = caption_ids.shape[1]  # update
                if args.save_attns:
                    output_attention_scores = output_attention_scores[1:]
            if outputs_per_token_list[-1] == '</s>':
                outputs_per_token_list = outputs_per_token_list[:-1]
                caption_ids = caption_ids[:, :-1]
                scores_tuple = scores_tuple[:-1]
                output_caption_len = caption_ids.shape[1]  # update
                if args.save_attns:
                    output_attention_scores = output_attention_scores[:-1]
                    
            intact_scores = torch.cat(scores_tuple, dim=0)
            if args.save_attns:
                first_token_attn_scores = output_attention_scores[0]  # tuple, length=32=num_layers
                # print(len(first_token_attn_scores))
                print(first_token_attn_scores[0].shape)  # TODO analysize first token for now [1, 32=num_attention_heads, 1or647, 648=prev_num_tokens]
                attn_file_name = args.answers_file_name.replace('.jsonl', f'-attn-scores_{idx}.hdf5')
                attn_h5py_file = os.path.expanduser(os.path.join(args.result_path, attn_file_name))
                attn_res_file = h5py.File(attn_h5py_file, 'w')
                for layer_idx, attn_scores in enumerate(first_token_attn_scores):
                    
                    attn_res_file.create_dataset(idx+'_'+str(layer_idx), 
                                            (attn_scores.shape[0], attn_scores.shape[1],
                                            attn_scores.shape[2], attn_scores.shape[3]), 
                                            data=attn_scores.cpu().numpy())
            
            if args.num_beams == 1:
                assert output_caption_len == intact_scores.shape[0]
            assert len(outputs_per_token_list) == output_caption_len
            
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
                                    "caption_tokens": outputs_per_token_list,
                                    "model_id": model_name,
                                    "image_path": image_path,
                                    "num_vis_tokens": num_vis_tokens,
                                    "mm_vision_select_feature":args.mm_vision_select_feature,
                                    "image_aspect_ratio":args.image_aspect_ratio,
                                    #    "mm_patch_merge_type":args.mm_patch_merge_type,
                                    "mm_vision_reduction_scale":args.mm_vision_reduction_scale,
                                    "min_single_token_dup_num":args.min_single_token_dup_num,
                                    }) + "\n")
            ans_file.flush()
            if args.save_logits:
                intact_scores_to_save = intact_scores[:1, :]
                logits_file.create_dataset(str(idx), 
                                        (intact_scores_to_save.shape[0], intact_scores_to_save.shape[1]), 
                                        data=intact_scores_to_save.cpu().numpy())
        ans_file.close()
        print(f"mm_vision_reduction_scale = {args.mm_vision_reduction_scale} finished")
    print(f"\nall finished\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--coco_anno_path", type=str, default="")
    parser.add_argument("--coco_split", type=list, default=[])
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--answers_file_name", type=str, default=None)
    parser.add_argument("--save_logits", type=bool, default=False)
    parser.add_argument("--logits_file_name", type=str, default=None)
    
    parser.add_argument("--conv_mode", type=str, default="llava_v1")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--alpha_noise", type=float, default=1.0)
    parser.add_argument("--alpha_nns", type=float, default=1.0)
    parser.add_argument("--alpha_base", type=float, default=1.0)
    parser.add_argument("--jsd_thres", type=float, default=None)
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
    
    # args.model_path = '/apdcephfs_gy4_303464260/share_303464260/dichyang/opensource_models/llava-v1.5-7b'
    args.model_path = '/apdcephfs_gy4_303464260/share_303464260/dichyang/opensource_models/llava-v1.5-13b'  # TODO
    
    # args.n_list = [1, 2, 3, 4, 5, 6, 7, 8,
    #                 9, 10, 11, 12, 13, 14, 15, 16, 
    #                 17, 18, 19, 20, 21, 22, 23, 24,
    #                 37, 38, 39, 40, 41, 42, 43, 44,
    #                 73, 74, 75, 76, 77, 78, 79, 80,
    #                 145, 146, 147, 148, 149, 150, 151, 152,
    #                 217, 218, 219, 220, 221, 222, 223, 224,
    #                 289, 290, 291, 292, 293, 294, 295, 296,
    #                 361, 362, 363, 364, 365, 366, 367, 368,
    #                 433, 434, 435, 436, 437, 438, 439, 440,
    #                 505, 506, 507, 508, 509, 510, 511, 512,
    #                 569, 570, 571, 572, 573, 574, 575, 576]
    args.n_list = [0]  # TODO
    
    args.coco_anno_path = '/apdcephfs_cq8/share_1367250/dichyang/code/Projects/VCD/experiments/data/cococap/annotations/dataset_coco.json'
    args.coco_path = '/apdcephfs_cq8/share_1367250/dichyang/data/coco/coco_for_token_reduce_exp/images/'
    args.result_path_template = '/apdcephfs_cq8/share_1367250/dichyang/data/coco/coco_for_token_reduce_exp/results_part2/'  # TODO
    
    args.coco_split = 'train'
    args.image_content = 'intact'
    args.decode_method = 'greedy'
    
    args.save_logits = True  # TODO
    args.save_attns = False  # TODO
    args.save_visual_features = False  # TODO
    
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
    
    # args.repetition_penalty = 1.0  # DO NOT support logit postprocessor as llava use -200 image token
        
    this_config = json.load(open(os.path.join(args.model_path, "config.json")))
    args.mm_vision_select_feature = this_config["mm_vision_select_feature"]
    args.image_aspect_ratio = this_config["image_aspect_ratio"]
    # args.mm_patch_merge_type = this_config["mm_patch_merge_type"]
    args.min_single_token_dup_num = this_config["min_single_token_dup_num"]
    if args.mm_vision_select_feature in ['low_sim_patch', 'high_sim_patch', 
                                         'fix_sim_patch', # important, single input
                                         'fix_sim_patch_pad', 
                                         'fix_sim_context_pad_patch',  # important, regional leave-one-out
                                         "fix_sim_context_tgt_patch",  # important, regional leave-one-out
                                         "fix_sim_context_patch",
                                         "fix_sim_ablate_context_tgt_patch",  # important, global leave-one-patch-out
                                         "fix_sim_ablate_tgt_patch",
                                         ]:
        args.mm_vision_reduction_scale = this_config["mm_vision_reduction_scale"]
    else:
        args.mm_vision_reduction_scale = 0
    
    args.answer_file_prefix = 'llava15_coco_zeroshot_captions_token_reduc_pre_exp_vitcls_ppl'
    
    set_seed(args.seed)
    eval_model(args)
    