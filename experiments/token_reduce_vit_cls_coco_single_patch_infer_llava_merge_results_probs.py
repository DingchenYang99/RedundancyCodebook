import sys
import argparse
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
# from sklearn.manifold import TSNE
from pathlib import Path
import json
import h5py
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.utils import disable_torch_init
# from lavis.models import load_model_and_preprocess
from transformers import AutoTokenizer, LlamaTokenizer
from data.whoops.whoops_utils import get_timestamp
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = '/apdcephfs_gy4_303464260/share_303464260/dichyang/opensource_models/llava-v1.5-13b'  # TODO
    model_path = os.path.expanduser(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    this_config = json.load(open(os.path.join(model_path, "config.json")))
    mm_use_im_start_end = getattr(this_config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(this_config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        
    res_root = '/apdcephfs_cq8/share_1367250/dichyang/data/coco/coco_for_token_reduce_exp/results/'  # TODO
    fs_dir = res_root + get_timestamp()
    Path(fs_dir).mkdir(parents=True, exist_ok=True)
    
    coco_split = 'train'  # TODO
    date_dirs = [
            "250515-135352",
            "250515-135446",
            "250515-135538",
            "250515-135630",
            "250515-135721",
            "250515-135813",
            "250515-135905",
            "250515-135957",
            "250515-140056",
            "250515-140153",
            "250515-140246",
            "250515-140338",
            "250515-140429",
            "250515-140521",
            "250515-140613",
            "250515-140705",
            "250515-140757",
            "250515-140850",
            "250515-140943",
            "250515-141035",
            "250515-141129",
            "250515-141226",
            "250515-141319",
            "250515-141412",
            "250515-141504",
            "250515-141557",
            "250515-141652",
            "250515-141745",
            "250515-141840",
            "250515-141934",
            "250515-142032",
            "250515-142127",
            "250515-142220",
            "250515-142313",
            "250515-142406",
            "250515-142459",
            "250515-142552",
            "250515-142646",
            "250515-142739",
            "250515-142832",
            "250515-142925",
            "250515-143018",
            "250515-143110",
            "250515-143202",
            "250515-143254",
            "250515-143346",
            "250515-143439",
            "250515-143531",
            "250515-143623",
            "250515-143716",
            "250515-143808",
            "250515-143900",
            "250515-143953",
            "250515-144046",
            "250515-144139",
            "250515-144232",
            "250515-144325",
            "250515-144418",
            "250515-144512",
            "250515-144606",
            "250515-144700",
            "250515-144755",
            "250515-144849",
            "250515-144944",
            "250515-145038",
            "250515-145132",
            "250515-145226",
            "250515-145329",
            "250515-145423",
            "250515-145517",
            "250515-145610",
            "250515-145704",
            "250515-145802",
            "250515-145856",
            "250515-145950",
            "250515-150044",
            "250515-150158",
            "250515-150251",
            "250515-150345",
            "250515-150439",
            "250515-150532",
            "250515-150633",
            "250515-150737",
            "250515-150841",
            "250515-150944",
            "250515-151048",
            "250515-151142",
            "250515-151235",
            "250515-151329",
            "250515-151422",
            "250515-151514",
            "250515-151606",
            "250515-151658",
            "250515-151750",
            "250515-151842",
            "250515-151935",
        ]  # TODO
    topk = 50  # head vocab, TODO
    
    fs_path = fs_dir + f'/llava15_13B-coco-zeroshot-captions-token-reduce-vitcls-fix-sim-patch-135352-151935_result.jsonl'  # TODO
    if os.path.exists(fs_path):
        print(f"{fs_path} already exists, please check!")
        raise ValueError
    
    fs = open(fs_path, "w", encoding='utf-8')
    
    all_samples_dict = {}
    for date_dir in date_dirs:
        fr_file_names = [q for q in os.listdir(os.path.join(res_root,date_dir)) if q.endswith(".jsonl")]
        assert len(fr_file_names) == 1
        fr_file_name = fr_file_names[0]
        result_file_name = os.path.join(res_root,date_dir,fr_file_name)
        fix_patch_sim_idx = int(fr_file_name.split(".")[0].split("-")[-1])
        # logits_file_name = result_file_name.replace(f"{fix_patch_sim_idx}.jsonl", f"{1}.hdf5")
        logits_file_name = result_file_name.replace(f".jsonl", ".hdf5") # fixed
        fr = (json.loads(q) for q in open(result_file_name, 'r', encoding='utf-8'))
        scores = h5py.File(os.path.expanduser(logits_file_name), 'r')
        
        for datasample in tqdm(fr):
            sample_id = datasample["image_id"]
            if sample_id not in all_samples_dict:
                all_samples_dict[sample_id] = {}
                
            caption_pred = datasample["caption_pred"]
            caption_tokens = datasample["caption_tokens"]
            image_path = datasample["image_path"]
            logits_pred = torch.tensor(scores[sample_id][()]).to(device)  # [num_tokens, num_vocab]
            assert logits_pred.ndim == 2
            assert logits_pred.shape[0] == 1  # len(caption_tokens)
            logits_pred_first_token = logits_pred[0, :]
            assert logits_pred_first_token.ndim == 1
            topk_logits_first_token, topk_idx_first_token = torch.topk(logits_pred_first_token, k=topk)
            # first_token_head_vocab_tokens = []
            for i in range(topk_idx_first_token.shape[-1]):
                this_candidate_token = tokenizer.convert_ids_to_tokens(topk_idx_first_token.unsqueeze(0)[:, i:i+1])[0]
                if i == 0:
                    assert this_candidate_token == caption_tokens[0]
                    break
                # print(this_candidate_token)
                # first_token_head_vocab_tokens.append(this_candidate_token)
            
            topk_probs_first_token = nn.functional.softmax(topk_logits_first_token, dim=-1)
            assert topk_probs_first_token[0].item() == (max_prob_first_token := topk_probs_first_token.max().item())
            # max_prob_first_token = topk_probs_first_token.max().item()
            assert fix_patch_sim_idx not in all_samples_dict[sample_id]
            all_samples_dict[sample_id].update({fix_patch_sim_idx:max_prob_first_token})
    
    print(f"writing to disk...")
    for sample_id, v in all_samples_dict.items():
        sorted_v = {key: v[key] for key in sorted(v.keys())}
        # print(sample_id, sorted_v)
        fs.write(json.dumps({
                "sample_id":sample_id,
                "fix_patch_sim_idx_and_max_prob_first_token":sorted_v,
                "top-k_head_vocab":topk,
            })+ "\n")
        fs.flush()
    
    fs.close()
    print(f"finished")