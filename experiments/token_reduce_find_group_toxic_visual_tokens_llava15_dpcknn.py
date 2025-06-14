import sys
import argparse
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
from pathlib import Path
from data.whoops.whoops_utils import get_timestamp
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from eval_utils.dpc_cluster import cluster_dpc_knn
import json
import h5py

all_date_dirs = [
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


if __name__ == '__main__':
    
    data_root = '/apdcephfs_cq8/share_1367250/dichyang/data/coco/coco_for_token_reduce_exp/results/'  # TODO
    data_root_2 = '/apdcephfs_cq8/share_1367250/dichyang/data/coco/coco_for_token_reduce_exp/results_part2/'  # TODO
    coco_split = 'train'
    result_path = data_root + get_timestamp() 
    Path(result_path).mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    top1_prob_thres = 0.077  # TODO
    cluster_nums = 8  # TODO
    head_vocab_topk = 50  # TODO
    cluster_size_thres = 8  # TODO
    # interested_sample_id = '520208'  # TODO
    use_jsd_thres = True  # TODO
    prototype_cluster_nums = 64 # TODO
    prototype_min_unique_sample_thres = 64  # TODO
    if use_jsd_thres:
        jsd_result_file_name_context = data_root + '250515-230457-results/check_jsd-fix_sim_context_tgt_patch_vs_fix_sim_context_pad_patch_all_patch-181423-215423-topk-10_result.jsonl'  # TODO
        jsd_result_file_name_global = data_root_2 + '250516-001438-results/check_jsd-fix_sim_ablate_context_tgt_patch_vs_global_all_patch-201941-001144_topk-10_result.jsonl'  # TODO
        jsd_value_thres = 1.8e-3  # TODO
        k_context = 1  # TODO
        k_global = 16  # TODO
    
    samples_visual_tokens_dict = {}
    samples_logits_dict = {}
    total_visual_token_num = 0
    total_logits_num = 0
    all_toxic_visual_tokens_num = 0
    
    print(f"reading single-forward visual tokens and logits...")
    for date_dir in tqdm(all_date_dirs):
        # visual tokens
        fr_file_names = [q for q in os.listdir(os.path.join(data_root, date_dir)) if q.endswith(".pt")]
        # corresponding logits
        logits_file_names = [q for q in os.listdir(
                os.path.join(data_root,date_dir)) if q.endswith(".hdf5")]
        assert len(logits_file_names) == 1
        logits_file_name = os.path.join(data_root,date_dir,logits_file_names[0])
        scores = h5py.File(os.path.expanduser(logits_file_name), 'r')
        # this_score_id = int(logits_file_name.split(".")[0].split("-")[-1])
        
        for fr_file_name in fr_file_names:
            sample_id = str(int(fr_file_name.split("_")[2]))
            
            # if sample_id != interested_sample_id:
                # continue  # TODO
                
            # visual tokens
            if sample_id not in samples_visual_tokens_dict:
                samples_visual_tokens_dict[sample_id] = {}
                
            fix_patch_sim_idx = int(fr_file_name.split("_")[-2].split("-")[-1])
            # assert this_score_id == fix_patch_sim_idx
            # print(f"extracting {sample_id}'s toxic visual token {fix_patch_sim_idx}...")
            
            result_file_name = os.path.join(data_root,date_dir,fr_file_name)
            visual_tokens = torch.load(result_file_name).to(device)
            # print(visual_tokens.shape)
            assert visual_tokens.shape[0] == 1
            visual_tokens = visual_tokens.squeeze(0)
            num_visual_tokens = visual_tokens.shape[0]
            # all_toxic_visual_tokens_num += num_visual_tokens
            # all_toxic_visual_tokens.append(visual_tokens)
            assert fix_patch_sim_idx not in samples_visual_tokens_dict[sample_id]
            samples_visual_tokens_dict[sample_id].update({
                fix_patch_sim_idx:visual_tokens
            })
            total_visual_token_num += num_visual_tokens
            
            # logits
            logits_pred = torch.tensor(scores[sample_id][()]).to(device)
            if sample_id not in samples_logits_dict:
                samples_logits_dict[sample_id] = {}
            assert logits_pred.ndim == 2
            logits_pred_first_token = logits_pred[0, :]
            assert len(logits_pred_first_token.shape) == 1
            
            topk_logits_first_token, topk_idx_first_token = torch.topk(logits_pred_first_token, k=head_vocab_topk)
            topk_probs_first_token = nn.functional.softmax(topk_logits_first_token, dim=-1)
            max_prob_first_token = topk_probs_first_token.max().item()
            assert fix_patch_sim_idx not in samples_logits_dict[sample_id]
            samples_logits_dict[sample_id].update({
                fix_patch_sim_idx:max_prob_first_token
            })
            total_logits_num += 1
            
    print(f"{len(samples_visual_tokens_dict.keys())} samples' {total_visual_token_num} visual tokens loaded")
    print(f"{len(samples_logits_dict.keys())} samples' {total_logits_num} logits loaded")
    
    if use_jsd_thres:
        # context
        print(f"reading context leave-one-out jsd values...")
        assert os.path.exists(jsd_result_file_name_context), f'invalid context jsd result file name {jsd_result_file_name_context}'
        context_jsd_result_dict = {}
        context_jsd = [json.loads(q) for q in open(jsd_result_file_name_context, 'r', encoding='utf-8')]
        for datasample_jsd in context_jsd:
            assert len(datasample_jsd.keys()) == 1
            for k_jsd in datasample_jsd.keys():
                assert k_jsd not in context_jsd_result_dict
            context_jsd_result_dict.update(datasample_jsd)
        
        # global
        print(f"loaded {len(context_jsd_result_dict.keys())} samples with context jsd value")
        assert jsd_result_file_name_global != jsd_result_file_name_context
        print(f"reading global leave-one-out jsd values...")
        assert os.path.exists(jsd_result_file_name_global), f'invalid global jsd result file name {jsd_result_file_name_global}'
        global_jsd_result_dict = {}
        global_jsd = [json.loads(q) for q in open(jsd_result_file_name_global, 'r', encoding='utf-8')]
        for datasample_jsd in global_jsd:
            assert len(datasample_jsd.keys()) == 1
            for k_jsd in datasample_jsd.keys():
                assert k_jsd not in global_jsd_result_dict
            global_jsd_result_dict.update(datasample_jsd)
        print(f"loaded {len(global_jsd_result_dict.keys())} samples with global jsd value")

    print(f"running dpc-knn for each sample...")
    all_low_infofeats_list_selected = []
    sample_id_store = []
    for sample_id, fix_patch_sim_idx_visual_tokens in tqdm(samples_visual_tokens_dict.items()):
        this_sample_logits = samples_logits_dict[sample_id]
        if use_jsd_thres:
            this_sample_jsds_context = context_jsd_result_dict[sample_id]
            this_sample_jsds_global = global_jsd_result_dict[sample_id]
            
        this_feats_list = []
        this_patch_ids_low_info_list = []
        is_low_info_patch_flag_list = []
        
        for this_patch_idx, this_visual_tokens in fix_patch_sim_idx_visual_tokens.items():
            this_this_visual_tokens_num = this_visual_tokens.shape[0]
            this_max_prob = this_sample_logits[this_patch_idx]
            if use_jsd_thres:
                this_jsd_value_context = this_sample_jsds_context[str(this_patch_idx)]
                this_jsd_value_global = this_sample_jsds_global[str(this_patch_idx)]
                this_jsd_value = k_context * this_jsd_value_context + k_global * this_jsd_value_global
                if (this_jsd_value < jsd_value_thres) and (this_max_prob < top1_prob_thres):
                    is_low_info_patch_flag_list.extend([True]*this_this_visual_tokens_num)
                else:
                    is_low_info_patch_flag_list.extend([False]*this_this_visual_tokens_num)
            else:
                # if this_max_prob > top1_prob_thres:
                # continue  # informative patch
                if this_max_prob < top1_prob_thres:
                    is_low_info_patch_flag_list.extend([True]*this_this_visual_tokens_num)
                else:
                    is_low_info_patch_flag_list.extend([False]*this_this_visual_tokens_num)
            
            # calculate dpc-knn on low-info patches only
            this_feats_list.append(this_visual_tokens)
            this_patch_ids_low_info_list.extend([this_patch_idx]*this_this_visual_tokens_num)
            
        print(f"{len(this_feats_list)} patches for {sample_id} filtered")
        this_feats_low_info = torch.cat(this_feats_list, dim=0)
        ll = this_feats_low_info.shape[0]
        # dpc-knn for each sample
        idx_cluster, _ = cluster_dpc_knn(
            token_dict={"x":this_feats_low_info[None, ...]},
            cluster_num=cluster_nums,
        )
        idx_cluster = idx_cluster.squeeze(0).tolist()
        assert len(idx_cluster) == ll, f"{ll}, {len(idx_cluster)}"
        assert len(is_low_info_patch_flag_list) == ll
        assert len(this_patch_ids_low_info_list) == ll
        cluster_idx_lab_dict = {}
        # cluster_is_low_lab_dict = {}
        for this_patch_ids, clus_label, is_low_info_flag_ in zip(this_patch_ids_low_info_list, 
                                                                 idx_cluster,
                                                                 is_low_info_patch_flag_list):
            # if not is_low_info_flag_:
                # continue
            if clus_label not in cluster_idx_lab_dict:
                cluster_idx_lab_dict[clus_label] = {
                    "is_low_info_flag_list":[],
                    "patch_ids_list":[],
                }
            cluster_idx_lab_dict[clus_label]["patch_ids_list"].append(this_patch_ids)
            cluster_idx_lab_dict[clus_label]["is_low_info_flag_list"].append(is_low_info_flag_)
        
        low_info_patch_ids_selected = []
        for k_, v_ in cluster_idx_lab_dict.items():
            # print(cluster_idx_lab_dict)
            # print(f"{k_}:\t{v_}")
            if len(v_["patch_ids_list"]) <= cluster_size_thres:
                for v_is_low_info_flag, v_patch_ids in zip(v_["is_low_info_flag_list"],v_["patch_ids_list"]):
                    if v_is_low_info_flag:
                        low_info_patch_ids_selected.append(v_patch_ids)
                        # print(f"{k_}:\t{v_patch_ids}")
                # low_info_patch_ids_selected.extend(v_)
        this_low_infofeats_list_selected = [fix_patch_sim_idx_visual_tokens[i] for i in low_info_patch_ids_selected]
        if not this_low_infofeats_list_selected or len(this_low_infofeats_list_selected) < 1:
            print(f"{sample_id} has NONE low-info patches, skipped")
            continue
        this_low_infofeats_selected = torch.cat(this_low_infofeats_list_selected, dim=0)
        all_toxic_visual_tokens_num += (lll := this_low_infofeats_selected.shape[0])
        all_low_infofeats_list_selected.append(this_low_infofeats_selected)
        sample_id_store.extend([sample_id]*lll)
    
    all_low_infofeats_selected = torch.cat(all_low_infofeats_list_selected, dim=0)
    
    print(f"running dpc-knn for selected visual tokens...")
    print(all_low_infofeats_selected.shape)
    assert all_low_infofeats_selected.shape[0] == len(sample_id_store)
    # dpc-knn for all low-info visual tokens
    prototype_idx_cluster, _ = cluster_dpc_knn(
        token_dict={"x":all_low_infofeats_selected[None, ...]},
        cluster_num=prototype_cluster_nums,
    )
    prototype_idx_cluster = prototype_idx_cluster.squeeze(0).tolist()
    assert len(prototype_idx_cluster) == len(sample_id_store)
    prototype_cluster_idx_lab_dict = {}
    for j, (prototype_sample_id, prototype_clus_label) in enumerate(zip(sample_id_store, prototype_idx_cluster)):
        if prototype_clus_label not in prototype_cluster_idx_lab_dict:
            prototype_cluster_idx_lab_dict[prototype_clus_label] = {
                    "sample_ids_list":[],
                    "prototype_visual_tokens":[],
                }
        prototype_cluster_idx_lab_dict[prototype_clus_label]["sample_ids_list"].append(prototype_sample_id)
        prototype_cluster_idx_lab_dict[prototype_clus_label]["prototype_visual_tokens"].append(
            all_low_infofeats_selected[j:j+1, ...])
    prototype_selected = []
    sample_id_store_filtered = []
    for k_proto, v_proto in prototype_cluster_idx_lab_dict.items():
        print(f"cluster: {k_proto}, samples:", v_proto["sample_ids_list"])
        num_nuique_samples_in_clus = len(set(v_proto["sample_ids_list"]))
        if num_nuique_samples_in_clus <= prototype_min_unique_sample_thres:
            continue
        prototype_selected.extend(v_proto["prototype_visual_tokens"])
        sample_id_store_filtered.extend(v_proto["sample_ids_list"])
        
    if len(prototype_selected) == 0:
        print("did not find any toxic prototypes!!")
        raise ValueError
    filtered_all_low_infofeats_selected = torch.cat(prototype_selected, dim=0)
    print(filtered_all_low_infofeats_selected.shape)
    all_toxic_visual_tokens_num = filtered_all_low_infofeats_selected.shape[0]
    assert all_toxic_visual_tokens_num == len(sample_id_store_filtered)
    
    print("save to disk...")
    result_save_path = os.path.join(result_path, f"coco_{coco_split}_500samples_numPrototypes-{all_toxic_visual_tokens_num}.pt")  # TODO
    torch.save(filtered_all_low_infofeats_selected, result_save_path)
    with open(result_save_path.replace(".pt", ".json"), 'w') as ffi:
        json.dump(sample_id_store_filtered, ffi)
    print(f"finished filtering {all_toxic_visual_tokens_num} toxic visual tokens")
    print(f"finished write to disk: {result_save_path}")