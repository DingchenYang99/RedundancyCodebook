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
# import torch.nn as nn
import torch.nn.functional as F
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

def calculate_jsd(base_logits, ref_logits):
    assert base_logits.shape == ref_logits.shape
    bs, d = base_logits.shape
    assert bs == 1, 'jensen shannon divergence control support bs=1 only for now'
    base_logits_smx = F.softmax(base_logits, dim=-1)
    ref_logits_smx = F.softmax(ref_logits, dim=-1)
    M = 0.5 * (base_logits_smx + ref_logits_smx)
    base_logits_logsmx = F.log_softmax(base_logits, dim=-1)
    ref_logits_logsmx = F.log_softmax(ref_logits, dim=-1)
    
    kl1 = F.kl_div(base_logits_logsmx, M, reduction='none').mean(-1)
    kl2 = F.kl_div(ref_logits_logsmx, M, reduction='none').mean(-1)
    # shape: (bs, )
    js_divs = 0.5 * (kl1 + kl2)
    # js_divs = js_divs.mean(-1)
    # print(js_divs)
    return js_divs

date_dirs_context_tgt = [
        # === fix_sim_context_tgt_patch ===
        # "250515-201140",
        # "250515-201244",
        # "250515-201348",
        # "250515-201452",
        # "250515-201555",
        # "250515-201658",
        # "250515-201802",
        # "250515-201905",
        # "250515-202009",
        # "250515-202112",
        # "250515-202216",
        # "250515-202319",
        # "250515-202424",
        # "250515-202527",
        # "250515-202631",
        # "250515-202735",
        # "250515-202839",
        # "250515-202944",
        # "250515-203048",
        # "250515-203152",
        # "250515-203256",
        # "250515-203401",
        # "250515-203506",
        # "250515-203611",
        # "250515-203716",
        # "250515-203821",
        # "250515-203925",
        # "250515-204031",
        # "250515-204136",
        # "250515-204241",
        # "250515-204347",
        # "250515-204452",
        # "250515-204558",
        # "250515-204702",
        # "250515-204807",
        # "250515-204912",
        # "250515-205017",
        # "250515-205122",
        # "250515-205226",
        # "250515-205330",
        # "250515-205435",
        # "250515-205539",
        # "250515-205643",
        # "250515-205747",
        # "250515-205850",
        # "250515-205954",
        # "250515-210057",
        # "250515-210200",
        # "250515-210304",
        # "250515-210409",
        # "250515-210513",
        # "250515-210617",
        # "250515-210722",
        # "250515-210826",
        # "250515-210932",
        # "250515-211036",
        # "250515-211141",
        # "250515-211246",
        # "250515-211352",
        # "250515-211458",
        # "250515-211605",
        # "250515-211711",
        # "250515-211817",
        # "250515-211923",
        # "250515-212028",
        # "250515-212134",
        # "250515-212239",
        # "250515-212345",
        # "250515-212449",
        # "250515-212555",
        # "250515-212700",
        # "250515-212807",
        # "250515-212913",
        # "250515-213019",
        # "250515-213124",
        # "250515-213230",
        # "250515-213336",
        # "250515-213441",
        # "250515-213547",
        # "250515-213653",
        # "250515-213758",
        # "250515-213905",
        # "250515-214010",
        # "250515-214115",
        # "250515-214220",
        # "250515-214325",
        # "250515-214431",
        # "250515-214537",
        # "250515-214644",
        # "250515-214750",
        # "250515-214856",
        # "250515-215001",
        # "250515-215107",
        # "250515-215212",
        # "250515-215317",
        # "250515-215423",
        ## === global 576 input ===
        # patch-0 500 smaples, *96
        "250516-001144",
    ]*96  # TODO

date_dirs_context_pad = [
        # # # === fix_sim_context_pad_patch ===
        # "250515-181423",
        # "250515-181617",
        # "250515-181721",
        # "250515-181825",
        # "250515-181928",
        # "250515-182031",
        # "250515-182134",
        # "250515-182239",
        # "250515-182343",
        # "250515-182447",
        # "250515-182551",
        # "250515-182655",
        # "250515-182758",
        # "250515-182901",
        # "250515-183004",
        # "250515-183108",
        # "250515-183211",
        # "250515-183314",
        # "250515-183418",
        # "250515-183521",
        # "250515-183625",
        # "250515-183729",
        # "250515-183834",
        # "250515-183940",
        # "250515-184045",
        # "250515-184151",
        # "250515-184256",
        # "250515-184401",
        # "250515-184506",
        # "250515-184612",
        # "250515-184717",
        # "250515-184822",
        # "250515-184926",
        # "250515-185031",
        # "250515-185135",
        # "250515-185239",
        # "250515-185343",
        # "250515-185447",
        # "250515-185551",
        # "250515-185654",
        # "250515-185758",
        # "250515-185902",
        # "250515-190005",
        # "250515-190109",
        # "250515-190212",
        # "250515-190316",
        # "250515-190419",
        # "250515-190524",
        # "250515-190628",
        # "250515-190733",
        # "250515-190839",
        # "250515-190944",
        # "250515-191049",
        # "250515-191154",
        # "250515-191259",
        # "250515-191404",
        # "250515-191510",
        # "250515-191616",
        # "250515-191721",
        # "250515-191826",
        # "250515-191933",
        # "250515-192038",
        # "250515-192143",
        # "250515-192248",
        # "250515-192354",
        # "250515-192500",
        # "250515-192617",
        # "250515-192806",
        # "250515-192955",
        # "250515-193143",
        # "250515-193332",
        # "250515-193522",
        # "250515-193714",
        # "250515-193856",
        # "250515-194003",
        # "250515-194110",
        # "250515-194218",
        # "250515-194325",
        # "250515-194432",
        # "250515-194538",
        # "250515-194645",
        # "250515-194753",
        # "250515-194900",
        # "250515-195006",
        # "250515-195112",
        # "250515-195219",
        # "250515-195326",
        # "250515-195433",
        # "250515-195541",
        # "250515-195649",
        # "250515-195755",
        # "250515-195901",
        # "250515-200008",
        # "250515-200113",
        # "250515-200219",
        # "250515-200326",
        # # === fix_sim_ablate_context_tgt_patch ===
        "250515-201941",
        "250515-202207",
        "250515-202432",
        "250515-202657",
        "250515-202922",
        "250515-203147",
        "250515-203412",
        "250515-203638",
        "250515-203903",
        "250515-204128",
        "250515-204353",
        "250515-204618",
        "250515-204843",
        "250515-205107",
        "250515-205331",
        "250515-205556",
        "250515-205820",
        "250515-210045",
        "250515-210309",
        "250515-210533",
        "250515-210758",
        "250515-211023",
        "250515-211247",
        "250515-211511",
        "250515-211736",
        "250515-212001",
        "250515-212225",
        "250515-212450",
        "250515-212714",
        "250515-212939",
        "250515-213203",
        "250515-213428",
        "250515-213652",
        "250515-213916",
        "250515-214141",
        "250515-214405",
        "250515-214630",
        "250515-214854",
        "250515-215118",
        "250515-215343",
        "250515-215608",
        "250515-215831",
        "250515-220055",
        "250515-220319",
        "250515-220543",
        "250515-220806",
        "250515-221030",
        "250515-221253",
        "250515-221517",
        "250515-221741",
        "250515-222049",
        "250515-222332",
        "250515-222556",
        "250515-222819",
        "250515-223044",
        "250515-223309",
        "250515-223533",
        "250515-223758",
        "250515-224022",
        "250515-224246",
        "250515-224511",
        "250515-224735",
        "250515-224959",
        "250515-225224",
        "250515-225448",
        "250515-225713",
        "250515-225937",
        "250515-230200",
        "250515-230424",
        "250515-230648",
        "250515-230913",
        "250515-231137",
        "250515-231401",
        "250515-231625",
        "250515-231850",
        "250515-232114",
        "250515-232339",
        "250515-232603",
        "250515-232826",
        "250515-233049",
        "250515-233313",
        "250515-233536",
        "250515-233800",
        "250515-234023",
        "250515-234247",
        "250515-234512",
        "250515-234735",
        "250515-234959",
        "250515-235222",
        "250515-235446",
        "250515-235709",
        "250515-235933",
        "250516-000157",
        "250516-000421",
        "250516-000645",
        "250516-000910",
    ]  # TODO

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
        
    interested_sample_id = 'all'  # 520208, or all TODO
    topk = 10  # head vocab, TODO
    
    res_root = '/apdcephfs_cq8/share_1367250/dichyang/data/coco/coco_for_token_reduce_exp/results_part2/'  # TODO part2
    fs_dir = res_root + get_timestamp()
    Path(fs_dir).mkdir(parents=True, exist_ok=True)
    
    # fs_path = fs_dir + f'/check_jsd-fix_sim_context_tgt_patch_vs_fix_sim_context_pad_patch_{interested_sample_id}_patch-181423-215423-topk-{topk}_result.jsonl'  # TODO
    fs_path = fs_dir + f'/check_jsd-fix_sim_ablate_context_tgt_patch_vs_global_all_patch-201941-001144_topk-10_result.jsonl'
    if os.path.exists(fs_path):
        print(f"{fs_path} already exists, please check!")
        raise ValueError
    fs = open(fs_path, "w", encoding='utf-8')
    
    assert len(date_dirs_context_tgt) == len(date_dirs_context_pad)
    result_dict = {}
    for date_dir_context_tgt, date_dir_context_pad in tqdm(zip(date_dirs_context_tgt, date_dirs_context_pad)):
        # date_dir_context_tgt = "241031-215546"
        # date_dir_context_pad = "241101-115851"  
        base_date_dir = date_dir_context_tgt
        
        fr_file_names_context_tgt = [q for q in os.listdir(os.path.join(res_root,date_dir_context_tgt)) if q.endswith(".jsonl")]
        fr_file_names_context_pad = [q for q in os.listdir(os.path.join(res_root,date_dir_context_pad)) if q.endswith(".jsonl")]
        
        assert len(fr_file_names_context_tgt) == 1 
        assert len(fr_file_names_context_pad) == 1
        fr_file_name_context_tgt = fr_file_names_context_tgt[0]
        fr_file_name_context_pad = fr_file_names_context_pad[0]
        
        assert 'fix_sim_context_tgt_patch' in fr_file_name_context_tgt or 'patch-0' in fr_file_name_context_tgt
        assert 'fix_sim_context_pad_patch' in fr_file_name_context_pad or 'fix_sim_ablate' in fr_file_name_context_pad
        # print(f"reading from {fr_file_name_context_tgt} and {fr_file_name_context_pad}...")
        fix_patch_sim_idx_context_tgt = int(fr_file_name_context_tgt.split(".")[0].split("-")[-1])
        fix_patch_sim_idx_context_pad = int(fr_file_name_context_pad.split(".")[0].split("-")[-1])
        if 'fix_sim_context' in fr_file_name_context_tgt and 'fix_sim_context' in fr_file_name_context_pad:
            assert fix_patch_sim_idx_context_tgt == fix_patch_sim_idx_context_pad
        fix_patch_sim_idx = fix_patch_sim_idx_context_tgt
        
        result_file_name_context_tgt = os.path.join(res_root,date_dir_context_tgt,fr_file_name_context_tgt)
        logits_file_name_context_tgt = result_file_name_context_tgt.replace(f".jsonl", f".hdf5")  # bug fixed
        fr_context_tgt = [json.loads(q) for q in open(result_file_name_context_tgt, 'r', encoding='utf-8')]
        scores_context_tgt = h5py.File(os.path.expanduser(logits_file_name_context_tgt), 'r')
        
        result_file_name_context_pad = os.path.join(res_root,date_dir_context_pad,fr_file_name_context_pad)
        logits_file_name_context_pad = result_file_name_context_pad.replace(".jsonl", ".hdf5")  # bug fixed
        fr_context_pad = [json.loads(q) for q in open(result_file_name_context_pad, 'r', encoding='utf-8')]
        scores_context_pad = h5py.File(os.path.expanduser(logits_file_name_context_pad), 'r')
        
        for datasample in fr_context_tgt:
            sample_id = datasample["image_id"]
            if interested_sample_id != 'all' and sample_id != interested_sample_id:
                continue
            if sample_id not in result_dict:
                result_dict[sample_id] = {}
            caption_pred = datasample["caption_pred"]
            caption_tokens = datasample["caption_tokens"]
            logits_pred_context_tgt = torch.tensor(scores_context_tgt[sample_id][()]).to(device)  # [num_tokens, num_vocab]
            logits_pred_context_pad = torch.tensor(scores_context_pad[sample_id][()]).to(device)  # [num_tokens, num_vocab]
            
            assert logits_pred_context_tgt.ndim == 2
            assert logits_pred_context_pad.ndim == 2
            
            assert logits_pred_context_tgt.shape[0] == 1 # len(caption_tokens)
            logits_pred_first_token_context_tgt = logits_pred_context_tgt[:1, :]  # [1, vocab_size]
            logits_pred_first_token_context_pad = logits_pred_context_pad[:1, :]
            # print(logits_pred_first_token_context_tgt.shape)
            # print(logits_pred_first_token_context_pad.shape)
            # topk_logits_first_token, topk_idx_first_token = torch.topk(logits_pred_first_token, k=topk)
            cutoffs, indices_to_keep = torch.topk(logits_pred_first_token_context_tgt, k=topk, dim=-1)
            logits_pred_first_token_context_tgt_topk = torch.gather(logits_pred_first_token_context_tgt, 
                                                                    dim=-1, index=indices_to_keep)
            logits_pred_first_token_context_pad_topk = torch.gather(logits_pred_first_token_context_pad, 
                                                                    dim=-1, index=indices_to_keep)
            # print(logits_pred_first_token_context_tgt_topk.shape)
            # print(logits_pred_first_token_context_pad_topk.shape)
            jsd_values = calculate_jsd(logits_pred_first_token_context_tgt_topk, logits_pred_first_token_context_pad_topk)
            jsd_values = jsd_values.item()
            
            # topk_logits_first_token_list_context_tgt = logits_pred_first_token_context_tgt_topk.squeeze(0).tolist()
            # topk_logits_first_token_list_context_pad = logits_pred_first_token_context_pad_topk.squeeze(0).tolist()
            # print(topk_logits_first_token_list_context_tgt)
            # print(topk_logits_first_token_list_context_pad)
            # print(f"interested_sample_id:{interested_sample_id}\tpatch:{fix_patch_sim_idx_context_tgt}\tjsd_value:{jsd_values}\n")
            # fs.write(f"interested_sample_id:{interested_sample_id}\tpatch:{fix_patch_sim_idx_context_tgt}\tjsd_value:{jsd_values}\n")
            assert fix_patch_sim_idx_context_pad not in result_dict[sample_id]
            result_dict[sample_id].update({fix_patch_sim_idx_context_pad:jsd_values})
    
    for sample_id_, ress in result_dict.items():
        this_data = {
            sample_id_:ress
        }
        fs.write(json.dumps(this_data, ensure_ascii=False)+"\n") 
        fs.flush()
    fs.close()
    print(f"finished")