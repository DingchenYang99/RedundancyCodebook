import torch
import torch.nn as nn
import math
from llava.utils import rank0_print
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

try:
    from s2wrapper import forward as multiscale_forward
except:
    pass


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")
        # costum
        self.save_cls_sim_file_name = "results.txt"  # TODO
        self.patch_retain_mask_bin = None
        self.patch_reduction_scale = getattr(args, 'mm_vision_reduction_scale', None)
        self.min_single_token_dup_num = getattr(args, 'min_single_token_dup_num', None)
        self.random_patch_num = getattr(args, 'random_patch_num', None)
        self.num_visual_tokens_buffer = 0
        self.toxic_codebook_flag = getattr(args, 'toxic_codebook_flag', False)
        if self.toxic_codebook_flag:
            self.toxic_codebook_path = getattr(args, 'toxic_codebook_path', None)
            self.toxic_visual_tokens = torch.load(self.toxic_codebook_path)
            self.toxic_codebook_thres = getattr(args, 'toxic_codebook_thres', 1)
            assert 0 <= self.toxic_codebook_thres <= 1
        
        self.pad_token_emb_path = getattr(args, 'pad_token_emb_path', None)
        self.pad_token_emb_flag = getattr(args, 'pad_token_emb_flag', None)
        if self.pad_token_emb_path and self.pad_token_emb_flag:
            self.pad_visual_tokens = torch.load(self.pad_token_emb_path)
            # self.retain_visual_token_idxs_in_pad = [int(q) for q in (getattr(args, 'retain_visual_token_idxs_in_pad', []))]
            assert 0 <= self.patch_reduction_scale <= 576
        
        if not delay_load:
            rank0_print(f"Loading vision tower: {vision_tower}")
            self.load_model()
        elif getattr(args, "unfreeze_mm_vision_tower", False):
            # TODO: better detector is needed.
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()
        elif hasattr(args, "mm_tunable_parts") and "mm_vision_tower" in args.mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        select_feature_type = self.select_feature

        if self.select_feature in ["slicefour_patch", "slicefour_cls_patch"]:
            select_every_k_layer = len(image_forward_outs.hidden_states) // 4
            image_features = torch.cat([image_forward_outs.hidden_states[i] for i in range(select_every_k_layer + self.select_layer, len(image_forward_outs.hidden_states), select_every_k_layer)], dim=-1)
            select_feature_type = select_feature_type.replace("slicefour_", "")
        elif self.select_feature in ["slice_m25811_f6_patch", "slice_m25811_f6_cls_patch"]:
            select_layers = [-2, -5, -8, -11, 6]
            image_features = torch.cat([image_forward_outs.hidden_states[i] for i in select_layers], dim=-1)
            select_feature_type = select_feature_type.replace("slice_m25811_f6_", "")
        else:
            image_features = image_forward_outs.hidden_states[self.select_layer]

        if select_feature_type == "patch":
            image_features = image_features[:, 1:]
            self.num_visual_tokens_buffer = image_features.shape[0] * image_features.shape[1]
        elif select_feature_type == "cls_patch":
            image_features = image_features
        # for token reduction
        elif select_feature_type == "random_patch":
            assert 0 < self.random_patch_num < 576, 'plz assign random_patch_num within (0,576) for llava-next'
            patch_features = image_features[:, 1:]
            bs, L, D = patch_features.shape
            sampled_L = self.random_patch_num
            keep_indices = torch.rand(bs, L).argsort(dim=1)[:, :sampled_L]
            keep_indices_sorted = keep_indices.sort(dim=1).values
            keep_patch_features = patch_features[torch.arange(bs).unsqueeze(1), keep_indices_sorted]
            image_features = keep_patch_features
            self.num_visual_tokens_buffer += image_features.shape[0] * image_features.shape[1]
        elif self.select_feature == 'low_sim_patch':
            patch_features = image_features[:, 1:]
            bs, L, D = patch_features.shape
            cls_token_norm = nn.functional.normalize(image_features[:, :1], p=2, dim=-1)
            patch_token_norm = nn.functional.normalize(patch_features, p=2, dim=-1)
            # cos_similarity = torch.bmm(cls_token_norm, patch_token_norm.transpose(1,2)).squeeze(1)
            cos_similarity = torch.matmul(patch_token_norm, cls_token_norm.transpose(1, 2)).squeeze(-1)  # [bs, L]
            # neg_cos_similarity = -cos_similarity
            # _, low_sim_index = torch.topk(neg_cos_similarity, k=int((L-1)*0.9), dim=-1, sorted=False)
            low_sim_index_bin = cos_similarity < torch.topk(
                cos_similarity, 
                k=int(L*self.patch_reduction_scale*0.01), 
                dim=-1)[0][..., -1:]  # [bs, L] fixed number of visual tokens  # TODO
            select_patch_feat_list = [patch_features[i][low_sim_index_bin[i]] for i in range(bs)]
            min_tokens_retain_in_split_img = min(select_patch_feat.shape[0] \
                for select_patch_feat in select_patch_feat_list)
            select_patch_feat_list_ = [select_patch_feat[:min_tokens_retain_in_split_img, ...].clone() \
                for select_patch_feat in select_patch_feat_list]
            image_features = torch.stack(select_patch_feat_list_)
            # image_features = torch.stack(select_patch_feat_list)
            # low_sim_index = torch.nonzero(low_sim_index_bin.float()).t()
            # low_sim_index = torch.nonzero(low_sim_index_bin.float())
            # image_features = torch.gather(image_features[:, 1:], dim=1, 
                                        #   index=low_sim_index.unsqueeze(-1).expand(-1,-1,D))
            # print(image_features.shape)
        elif self.select_feature == 'high_sim_patch':
            patch_features = image_features[:, 1:]
            bs, L, D = patch_features.shape
            cls_token_norm = nn.functional.normalize(image_features[:, :1], p=2, dim=-1)
            patch_token_norm = nn.functional.normalize(patch_features, p=2, dim=-1)
            # cos_similarity = torch.bmm(cls_token_norm, patch_token_norm.transpose(1,2)).squeeze(1)
            cos_similarity = torch.matmul(patch_token_norm, cls_token_norm.transpose(1, 2)).squeeze(-1)  # [bs, L]
            # neg_cos_similarity = -cos_similarity
            # _, low_sim_index = torch.topk(neg_cos_similarity, k=int((L-1)*0.9), dim=-1, sorted=False)
            low_sim_index_bin = cos_similarity > torch.topk(
                cos_similarity, 
                k=int(L*self.patch_reduction_scale*0.01), 
                dim=-1)[0][..., -1:]  # [bs, L] fixed number of visual tokens  # TODO
            select_patch_feat_list = [patch_features[i][low_sim_index_bin[i]] for i in range(bs)]
            min_tokens_retain_in_split_img = min(select_patch_feat.shape[0] \
                for select_patch_feat in select_patch_feat_list)
            select_patch_feat_list_ = [select_patch_feat[:min_tokens_retain_in_split_img, ...].clone() for select_patch_feat in select_patch_feat_list]
            image_features = torch.stack(select_patch_feat_list_)
            # low_sim_index = torch.nonzero(low_sim_index_bin.float()).t()
            # low_sim_index = torch.nonzero(low_sim_index_bin.float())
            # image_features = torch.gather(image_features[:, 1:], dim=1, 
                                        #   index=low_sim_index.unsqueeze(-1).expand(-1,-1,D))
            # print(image_features.shape)
        elif self.select_feature == 'fix_sim_patch':
            patch_features = image_features[:, 1:].detach().clone()
            # print(image_features.shape)
            bs, L, D = patch_features.shape
            assert bs == 1, "support bs=1 only for now for padding img tokens, do not split sub-images."
            cls_token_norm = nn.functional.normalize(image_features[:, :1].clone(), p=2, dim=-1)
            patch_token_norm = nn.functional.normalize(patch_features, p=2, dim=-1)
            # cos_similarity = torch.bmm(cls_token_norm, patch_token_norm.transpose(1,2)).squeeze(1)
            cos_similarity = torch.matmul(patch_token_norm, cls_token_norm.transpose(1, 2)).squeeze(-1)  # [bs, L]
            # neg_cos_similarity = -cos_similarity
            # _, low_sim_index = torch.topk(neg_cos_similarity, k=int((L-1)*0.9), dim=-1, sorted=False)
            topk_cos_similarity, topk_cos_similarity_idx = torch.topk(
                cos_similarity, 
                k=int(self.patch_reduction_scale),
                dim=-1)  # [bs, L] fixed number of visual tokens  # TODO
            
            low_sim_index_bin = cos_similarity == topk_cos_similarity[..., -1:]  # [bs, L] fixed number of visual tokens  # TODO
            # print(topk_cos_similarity_idx[0, -1].item())
            select_patch_feat_list = [patch_features[i][low_sim_index_bin[i]] for i in range(bs)]
            image_features = torch.stack(select_patch_feat_list)  # [bs, K, D]
            
            # method1 repeat
            image_features = image_features.repeat(1, self.min_single_token_dup_num, 1)
            
            # method2 padding
            # min_single_token_dup_num = self.min_single_token_dup_num
            # self.pad_visual_tokens = self.pad_visual_tokens.to(self.dtype).to(self.device)
            # dup_times = min_single_token_dup_num - image_features.shape[1]
            # pad_visual_tokens = self.pad_visual_tokens.clone()
            # pad_visual_tokens = pad_visual_tokens[None, ...].repeat(bs, dup_times, 1)
            # image_features = torch.cat([image_features, pad_visual_tokens], dim=1)
            
            # rubbish
            # low_sim_index = torch.nonzero(low_sim_index_bin.float()).t()
            # low_sim_index = torch.nonzero(low_sim_index_bin.float())
            # image_features = torch.gather(image_features[:, 1:], dim=1, 
                                        #   index=low_sim_index.unsqueeze(-1).expand(-1,-1,D))
            # print(image_features.shape)
            
        elif self.select_feature == 'fix_id_patch':
            patch_features = image_features[:, 1:].detach().clone()
            # print(image_features.shape)
            bs, L, D = patch_features.shape
            assert bs == 1, "support bs=1 only for now for padding img tokens, do not split sub-images."

            # print(topk_cos_similarity_idx[0, -1].item())
            select_patch_feat_list = [
                patch_features[i][int(self.patch_reduction_scale):int(self.patch_reduction_scale)+1, :] \
                    for i in range(bs)]
            image_features = torch.stack(select_patch_feat_list)  # [bs, K, D]
            # method1 repeat
            image_features = image_features.repeat(1, self.min_single_token_dup_num, 1)
        
        elif self.select_feature in ['high_info_patch', 'high_info_patch_flatten',
                                     'high_info_patch_double_smx', 'high_info_patch_single_smx',
                                     'high_info_patch_fix_num',
                                     ]:
            image_features = image_features[:, 1:]
            self.toxic_visual_tokens = self.toxic_visual_tokens.to(self.dtype).to(self.device)
        
        elif self.select_feature == 'fix_sim_patch_pad':
            raise NotImplementedError
            cls_features = image_features[:, :1]
            cls_token_norm = nn.functional.normalize(cls_features, p=2, dim=-1)
            
            patch_features = image_features[:, 1:]
            patch_token_norm = nn.functional.normalize(patch_features, p=2, dim=-1)
            # cos_similarity = torch.bmm(cls_token_norm, patch_token_norm.transpose(1,2)).squeeze(1)
            cos_similarity = torch.matmul(patch_token_norm, cls_token_norm.transpose(1, 2)).squeeze(-1)  # [bs, L]
            # neg_cos_similarity = -cos_similarity
            # _, low_sim_index = torch.topk(neg_cos_similarity, k=int((L-1)*0.9), dim=-1, sorted=False)
            _, topk_cos_similarity_idx = torch.topk(
                cos_similarity, 
                k=int(self.patch_reduction_scale),
                dim=-1)  # [bs, L] fixed number of visual tokens  # TODO
            
            self.retain_visual_token_idxs_in_pad = [topk_cos_similarity_idx[0, -1].item()]
            assert isinstance(self.retain_visual_token_idxs_in_pad, list)
            self.pad_visual_tokens = self.pad_visual_tokens.to(self.dtype).to(self.device)
            image_features = patch_features
        
        elif self.select_feature == 'fix_sim_context_tgt_patch':
            cls_features = image_features[:, :1].detach().clone()
            cls_token_norm = nn.functional.normalize(cls_features, p=2, dim=-1)
            
            patch_features = image_features[:, 1:].detach().clone()
            bs, L, D = patch_features.shape
            assert bs == 1, 'support bs=1 for now, 241030'
            patch_token_norm = nn.functional.normalize(patch_features, p=2, dim=-1)
            # cos_similarity = torch.bmm(cls_token_norm, patch_token_norm.transpose(1,2)).squeeze(1)
            cos_similarity = torch.matmul(patch_token_norm, cls_token_norm.transpose(1, 2)).squeeze(-1)  # [bs, L]
            # neg_cos_similarity = -cos_similarity
            # _, low_sim_index = torch.topk(neg_cos_similarity, k=int((L-1)*0.9), dim=-1, sorted=False)
            _, topk_cos_similarity_idx = torch.topk(
                cos_similarity, 
                k=int(self.patch_reduction_scale),
                dim=-1)  # [bs, L]
            
            tgt_visual_token_idxs = topk_cos_similarity_idx[0, -1].item()
            # print(f"center: {tgt_visual_token_idxs}")
            assert 0 <= tgt_visual_token_idxs < L
            # context level
            sqrt_L = math.sqrt(L)
            if not sqrt_L.is_integer():
                print(f"visual token number: {L} is invalid for math.sqrt()")
                raise ValueError
            h_num_tokens = w_num_tokens = int(sqrt_L)
            # llava-next resize the base image to square
            tgt_height_idx = tgt_visual_token_idxs // h_num_tokens
            tgt_width_idx = tgt_visual_token_idxs % w_num_tokens
            
            # 3x3 for now 241030
            if tgt_height_idx == 0:
                h_dirs = [0,1,2]
            elif tgt_height_idx == h_num_tokens-1:
                h_dirs = [-2,-1,0]
            else:
                h_dirs = [-1,0,1]
            
            if tgt_width_idx == 0:
                w_dirs = [0,1,2]
            elif tgt_width_idx == w_num_tokens-1:
                w_dirs = [-2,-1,0]
            else:
                w_dirs = [-1,0,1]
            
            nn_visual_token_list = []
            nn_visual_token_idxs_list = []
            for h_dir in h_dirs:
                for w_dir in w_dirs:
                    nn_visual_token_ids = (tgt_height_idx + h_dir) * h_num_tokens + (tgt_width_idx+w_dir)
                    # print(nn_visual_token_ids)
                    nn_visual_token_idxs_list.append(nn_visual_token_ids)
                    if h_dir == 0 and w_dir == 0:
                        assert nn_visual_token_ids == tgt_visual_token_idxs
                    this_nn_visual_token = patch_features[:1,nn_visual_token_ids:nn_visual_token_ids+1,:].clone()
                    nn_visual_token_list.append(this_nn_visual_token)
            # print(f"nns: {nn_visual_token_idxs_list}")
            assert (ll_nntokens := len(nn_visual_token_list)) == len(h_dirs) * len(w_dirs)
            patch_features_selected = torch.cat(nn_visual_token_list, dim=1)
            assert patch_features_selected.shape[1] == ll_nntokens
            image_features = patch_features_selected
        
        elif self.select_feature == 'fix_id_context_tgt_patch':
            patch_features = image_features[:, 1:].detach().clone()
            bs, L, D = patch_features.shape
            assert bs == 1, 'support bs=1 for now, 241030'
            
            tgt_visual_token_idxs = int(self.patch_reduction_scale)
            # print(f"center: {tgt_visual_token_idxs}")
            assert 0 <= tgt_visual_token_idxs < L
            # context level
            sqrt_L = math.sqrt(L)
            if not sqrt_L.is_integer():
                print(f"visual token number: {L} is invalid for math.sqrt()")
                raise ValueError
            h_num_tokens = w_num_tokens = int(sqrt_L)
            # llava-next resize the base image to square
            tgt_height_idx = tgt_visual_token_idxs // h_num_tokens
            tgt_width_idx = tgt_visual_token_idxs % w_num_tokens
            
            # 3x3 for now 241030
            if tgt_height_idx == 0:
                h_dirs = [0,1,2]
            elif tgt_height_idx == h_num_tokens-1:
                h_dirs = [-2,-1,0]
            else:
                h_dirs = [-1,0,1]
            
            if tgt_width_idx == 0:
                w_dirs = [0,1,2]
            elif tgt_width_idx == w_num_tokens-1:
                w_dirs = [-2,-1,0]
            else:
                w_dirs = [-1,0,1]
            
            nn_visual_token_list = []
            nn_visual_token_idxs_list = []
            for h_dir in h_dirs:
                for w_dir in w_dirs:
                    nn_visual_token_ids = (tgt_height_idx + h_dir) * h_num_tokens + (tgt_width_idx+w_dir)
                    # print(nn_visual_token_ids)
                    nn_visual_token_idxs_list.append(nn_visual_token_ids)
                    if h_dir == 0 and w_dir == 0:
                        assert nn_visual_token_ids == tgt_visual_token_idxs
                    this_nn_visual_token = patch_features[:1,nn_visual_token_ids:nn_visual_token_ids+1,:].clone()
                    nn_visual_token_list.append(this_nn_visual_token)
            # print(f"nns: {nn_visual_token_idxs_list}")
            assert (ll_nntokens := len(nn_visual_token_list)) == len(h_dirs) * len(w_dirs)
            patch_features_selected = torch.cat(nn_visual_token_list, dim=1)
            assert patch_features_selected.shape[1] == ll_nntokens
            image_features = patch_features_selected
            
        elif self.select_feature == 'fix_sim_context_pad_patch':
            assert self.pad_token_emb_path and self.pad_token_emb_flag, 'plz set pad_token_emb_flag to True'
            self.pad_visual_token = self.pad_visual_tokens.to(self.dtype).to(self.device)
            
            cls_features = image_features[:, :1].detach().clone()
            cls_token_norm = nn.functional.normalize(cls_features, p=2, dim=-1)
            
            patch_features = image_features[:, 1:].detach().clone()
            bs, L, D = patch_features.shape
            assert bs == 1, 'support bs=1 for now, 241030'
            patch_token_norm = nn.functional.normalize(patch_features, p=2, dim=-1)
            # cos_similarity = torch.bmm(cls_token_norm, patch_token_norm.transpose(1,2)).squeeze(1)
            cos_similarity = torch.matmul(patch_token_norm, cls_token_norm.transpose(1, 2)).squeeze(-1)  # [bs, L]
            # neg_cos_similarity = -cos_similarity
            # _, low_sim_index = torch.topk(neg_cos_similarity, k=int((L-1)*0.9), dim=-1, sorted=False)
            _, topk_cos_similarity_idx = torch.topk(
                cos_similarity, 
                k=int(self.patch_reduction_scale),
                dim=-1)  # [bs, L]
            
            tgt_visual_token_idxs = topk_cos_similarity_idx[0, -1].item()
            # print(f"center: {tgt_visual_token_idxs}")
            assert 0 <= tgt_visual_token_idxs < L
            # context level
            sqrt_L = math.sqrt(L)
            if not sqrt_L.is_integer():
                print(f"visual token number: {L} is invalid for math.sqrt()")
                raise ValueError
            h_num_tokens = w_num_tokens = int(sqrt_L)
            # llava-next resize the base image to square
            tgt_height_idx = tgt_visual_token_idxs // h_num_tokens
            tgt_width_idx = tgt_visual_token_idxs % w_num_tokens
            
            # 3x3 for now 241030
            if tgt_height_idx == 0:
                h_dirs = [0,1,2]
            elif tgt_height_idx == h_num_tokens-1:
                h_dirs = [-2,-1,0]
            else:
                h_dirs = [-1,0,1]
            
            if tgt_width_idx == 0:
                w_dirs = [0,1,2]
            elif tgt_width_idx == w_num_tokens-1:
                w_dirs = [-2,-1,0]
            else:
                w_dirs = [-1,0,1]
            
            nn_visual_token_list_prev = []
            nn_visual_token_list_post = []
            nn_visual_token_idxs_list = []
            for h_dir in h_dirs:
                for w_dir in w_dirs:
                    nn_visual_token_ids = (tgt_height_idx + h_dir) * h_num_tokens + (tgt_width_idx+w_dir)
                    # print(nn_visual_token_ids)
                    this_nn_visual_token = patch_features[:1,nn_visual_token_ids:nn_visual_token_ids+1,:].clone()
                    
                    if nn_visual_token_ids < tgt_visual_token_idxs:
                        nn_visual_token_list_prev.append(this_nn_visual_token)
                    
                    elif h_dir == 0 and w_dir == 0:
                        assert nn_visual_token_ids == tgt_visual_token_idxs
                        # nn_visual_token_list.append(pad_visual_token)
                        continue
                    
                    elif nn_visual_token_ids > tgt_visual_token_idxs:
                        nn_visual_token_list_post.append(this_nn_visual_token)
                    
                    else:
                        raise ValueError
                    
                    nn_visual_token_idxs_list.append(nn_visual_token_ids)
                    
            # print(f"nns: {nn_visual_token_idxs_list}")
            assert (ll_nntokens := (len(nn_visual_token_list_prev)+len(nn_visual_token_list_post))) \
                == len(h_dirs) * len(w_dirs) - 1
            if len(nn_visual_token_list_prev) > 0:
                patch_features_selected_prev = torch.cat(nn_visual_token_list_prev, dim=1)
            else:
                patch_features_selected_prev = None
                
            if len(nn_visual_token_list_post) > 0:
                patch_features_selected_post = torch.cat(nn_visual_token_list_post, dim=1)
            else:
                patch_features_selected_post = None
            # patch_features_selected = torch.cat(nn_visual_token_list, dim=1)
            # assert patch_features_selected.shape[1] == ll_nntokens
            # image_features = patch_features_selected
            image_features = [patch_features_selected_prev, patch_features_selected_post]
        
        elif self.select_feature == 'fix_id_context_pad_patch':
            assert self.pad_token_emb_path and self.pad_token_emb_flag, 'plz set pad_token_emb_flag to True'
            self.pad_visual_token = self.pad_visual_tokens.to(self.dtype).to(self.device)
            
            patch_features = image_features[:, 1:].detach().clone()
            bs, L, D = patch_features.shape
            assert bs == 1, 'support bs=1 for now, 241030'
            
            tgt_visual_token_idxs = int(self.patch_reduction_scale)
            # print(f"center: {tgt_visual_token_idxs}")
            assert 0 <= tgt_visual_token_idxs < L
            # context level
            sqrt_L = math.sqrt(L)
            if not sqrt_L.is_integer():
                print(f"visual token number: {L} is invalid for math.sqrt()")
                raise ValueError
            h_num_tokens = w_num_tokens = int(sqrt_L)
            # llava-next resize the base image to square
            tgt_height_idx = tgt_visual_token_idxs // h_num_tokens
            tgt_width_idx = tgt_visual_token_idxs % w_num_tokens
            
            # 3x3 for now 241030
            if tgt_height_idx == 0:
                h_dirs = [0,1,2]
            elif tgt_height_idx == h_num_tokens-1:
                h_dirs = [-2,-1,0]
            else:
                h_dirs = [-1,0,1]
            
            if tgt_width_idx == 0:
                w_dirs = [0,1,2]
            elif tgt_width_idx == w_num_tokens-1:
                w_dirs = [-2,-1,0]
            else:
                w_dirs = [-1,0,1]
            
            nn_visual_token_list_prev = []
            nn_visual_token_list_post = []
            nn_visual_token_idxs_list = []
            for h_dir in h_dirs:
                for w_dir in w_dirs:
                    nn_visual_token_ids = (tgt_height_idx + h_dir) * h_num_tokens + (tgt_width_idx+w_dir)
                    # print(nn_visual_token_ids)
                    this_nn_visual_token = patch_features[:1,nn_visual_token_ids:nn_visual_token_ids+1,:].clone()
                    
                    if nn_visual_token_ids < tgt_visual_token_idxs:
                        nn_visual_token_list_prev.append(this_nn_visual_token)
                    
                    elif h_dir == 0 and w_dir == 0:
                        assert nn_visual_token_ids == tgt_visual_token_idxs
                        # nn_visual_token_list.append(pad_visual_token)
                        continue
                    
                    elif nn_visual_token_ids > tgt_visual_token_idxs:
                        nn_visual_token_list_post.append(this_nn_visual_token)
                    
                    else:
                        raise ValueError
                    
                    nn_visual_token_idxs_list.append(nn_visual_token_ids)
                    
            # print(f"nns: {nn_visual_token_idxs_list}")
            assert (ll_nntokens := (len(nn_visual_token_list_prev)+len(nn_visual_token_list_post))) \
                == len(h_dirs) * len(w_dirs) - 1
            if len(nn_visual_token_list_prev) > 0:
                patch_features_selected_prev = torch.cat(nn_visual_token_list_prev, dim=1)
            else:
                patch_features_selected_prev = None
                
            if len(nn_visual_token_list_post) > 0:
                patch_features_selected_post = torch.cat(nn_visual_token_list_post, dim=1)
            else:
                patch_features_selected_post = None
            # patch_features_selected = torch.cat(nn_visual_token_list, dim=1)
            # assert patch_features_selected.shape[1] == ll_nntokens
            # image_features = patch_features_selected
            image_features = [patch_features_selected_prev, patch_features_selected_post]
            
        elif self.select_feature == 'fix_sim_context_patch':
            # raise NotImplementedError
            
            cls_features = image_features[:, :1].detach().clone()
            cls_token_norm = nn.functional.normalize(cls_features, p=2, dim=-1)
            
            patch_features = image_features[:, 1:].detach().clone()
            bs, L, D = patch_features.shape
            assert bs == 1, 'support bs=1 for now, 241030'
            patch_token_norm = nn.functional.normalize(patch_features, p=2, dim=-1)
            # cos_similarity = torch.bmm(cls_token_norm, patch_token_norm.transpose(1,2)).squeeze(1)
            cos_similarity = torch.matmul(patch_token_norm, cls_token_norm.transpose(1, 2)).squeeze(-1)  # [bs, L]
            # print(cos_similarity.shape)
            # neg_cos_similarity = -cos_similarity
            # _, low_sim_index = torch.topk(neg_cos_similarity, k=int((L-1)*0.9), dim=-1, sorted=False)
            topk_cos_similarity_value, topk_cos_similarity_idx = torch.topk(
                cos_similarity, 
                k=int(self.patch_reduction_scale),
                dim=-1)  # [bs, L]
            
            topk_cos_similarity_value_list = topk_cos_similarity_value[0].tolist()
            topk_cos_similarity_idx_list = topk_cos_similarity_idx[0].tolist()
            with open(self.save_cls_sim_file_name, "w") as ff:
                # print(self.save_cls_sim_file_name)
                # print(topk_cos_similarity_idx_list)
                # print(topk_cos_similarity_value_list)
                for i, (idxss_, valss_) in enumerate(zip(topk_cos_similarity_idx_list, topk_cos_similarity_value_list)):
                    # print(f"\"{idxss_}\": \"{valss_}\",")
                    # print("{" + f"\"{i+1}\"" + ": {" + f"\"{idxss_}\": {valss_}" + "}},")
                    line = "{" + f"\"{i+1}\"" + ": {" + f"\"{idxss_}\": {valss_}" + "}},\n"
                    # print(line)
                    ff.write(line)
                    # self.save_cls_sim_file_name.flush()
            image_features = image_features[:, 1:]
            # raise NotImplementedError
            # tgt_visual_token_idxs = topk_cos_similarity_idx[0, -1].item()
            # # print(f"center: {tgt_visual_token_idxs}")
            # assert 0 <= tgt_visual_token_idxs < L
            # # context level
            # sqrt_L = math.sqrt(L)
            # if not sqrt_L.is_integer():
            #     print(f"visual token number: {L} is invalid for math.sqrt()")
            #     raise ValueError
            # h_num_tokens = w_num_tokens = int(sqrt_L)
            # # llava-next resize the base image to square
            # tgt_height_idx = tgt_visual_token_idxs // h_num_tokens
            # tgt_width_idx = tgt_visual_token_idxs % w_num_tokens
            
            # # 3x3 for now 241030
            # if tgt_height_idx == 0:
            #     h_dirs = [0,1,2]
            # elif tgt_height_idx == h_num_tokens-1:
            #     h_dirs = [-2,-1,0]
            # else:
            #     h_dirs = [-1,0,1]
            
            # if tgt_width_idx == 0:
            #     w_dirs = [0,1,2]
            # elif tgt_width_idx == w_num_tokens-1:
            #     w_dirs = [-2,-1,0]
            # else:
            #     w_dirs = [-1,0,1]
            
            # nn_visual_token_list = []
            # for h_dir in h_dirs:
            #     for w_dir in w_dirs:
            #         nn_visual_token_ids = (tgt_height_idx + h_dir) * h_num_tokens + (tgt_width_idx+w_dir)
            #         # print(nn_visual_token_ids)
            #         if h_dir == 0 and w_dir == 0:
            #             assert nn_visual_token_ids == tgt_visual_token_idxs
            #             continue
            #         this_nn_visual_token = patch_features[:1,nn_visual_token_ids:nn_visual_token_ids+1,:].clone()
            #         nn_visual_token_list.append(this_nn_visual_token)
            # assert (ll_nntokens := len(nn_visual_token_list)) == len(h_dirs) * len(w_dirs) - 1
            # patch_features_selected = torch.cat(nn_visual_token_list, dim=1)
            # assert patch_features_selected.shape[1] == ll_nntokens
            # image_features = patch_features_selected
            
        elif self.select_feature == 'fix_sim_ablate_context_tgt_patch':
            assert self.pad_token_emb_path and self.pad_token_emb_flag, 'plz set pad_token_emb_flag to True'
            self.pad_visual_token = self.pad_visual_tokens.to(self.dtype).to(self.device)
            
            # get nn patch ids
            cls_features = image_features[:, :1].detach().clone()
            cls_token_norm = nn.functional.normalize(cls_features, p=2, dim=-1)
            
            patch_features = image_features[:, 1:].detach().clone()
            bs, L, D = patch_features.shape
            assert bs == 1, 'support bs=1 for now, 241030'
            patch_token_norm = nn.functional.normalize(patch_features, p=2, dim=-1)
            # cos_similarity = torch.bmm(cls_token_norm, patch_token_norm.transpose(1,2)).squeeze(1)
            cos_similarity = torch.matmul(patch_token_norm, cls_token_norm.transpose(1, 2)).squeeze(-1)  # [bs, L]
            # neg_cos_similarity = -cos_similarity
            # _, low_sim_index = torch.topk(neg_cos_similarity, k=int((L-1)*0.9), dim=-1, sorted=False)
            _, topk_cos_similarity_idx = torch.topk(
                cos_similarity, 
                k=int(self.patch_reduction_scale),
                dim=-1)  # [bs, L]
            
            tgt_visual_token_idxs = topk_cos_similarity_idx[0, -1].item()
            # print(f"center: {tgt_visual_token_idxs}")
            assert 0 <= tgt_visual_token_idxs < L
            # context level
            sqrt_L = math.sqrt(L)
            if not sqrt_L.is_integer():
                print(f"visual token number: {L} is invalid for math.sqrt()")
                raise ValueError
            h_num_tokens = w_num_tokens = int(sqrt_L)
            # llava-next resize the base image to square
            tgt_height_idx = tgt_visual_token_idxs // h_num_tokens
            tgt_width_idx = tgt_visual_token_idxs % w_num_tokens
            
            # 3x3 for now 241030
            if tgt_height_idx == 0:
                h_dirs = [0,1,2]
            elif tgt_height_idx == h_num_tokens-1:
                h_dirs = [-2,-1,0]
            else:
                h_dirs = [-1,0,1]
            
            if tgt_width_idx == 0:
                w_dirs = [0,1,2]
            elif tgt_width_idx == w_num_tokens-1:
                w_dirs = [-2,-1,0]
            else:
                w_dirs = [-1,0,1]
            
            self.nn_visual_token_idxs_list = []
            for h_dir in h_dirs:
                for w_dir in w_dirs:
                    nn_visual_token_ids = (tgt_height_idx + h_dir) * h_num_tokens + (tgt_width_idx+w_dir)
                    # print(nn_visual_token_ids)
                    self.nn_visual_token_idxs_list.append(nn_visual_token_ids)
            assert len(self.nn_visual_token_idxs_list) == len(h_dirs) * len(w_dirs)
            image_features = image_features[:, 1:]
        
        elif self.select_feature == 'fix_id_ablate_context_tgt_patch':
            assert self.pad_token_emb_path and self.pad_token_emb_flag, 'plz set pad_token_emb_flag to True'
            self.pad_visual_token = self.pad_visual_tokens.to(self.dtype).to(self.device)
            
            patch_features = image_features[:, 1:].detach().clone()
            bs, L, D = patch_features.shape
            assert bs == 1, 'support bs=1 for now, 241030'
            
            tgt_visual_token_idxs = int(self.patch_reduction_scale)
            # print(f"center: {tgt_visual_token_idxs}")
            assert 0 <= tgt_visual_token_idxs < L
            # context level
            sqrt_L = math.sqrt(L)
            if not sqrt_L.is_integer():
                print(f"visual token number: {L} is invalid for math.sqrt()")
                raise ValueError
            h_num_tokens = w_num_tokens = int(sqrt_L)
            # llava-next resize the base image to square
            tgt_height_idx = tgt_visual_token_idxs // h_num_tokens
            tgt_width_idx = tgt_visual_token_idxs % w_num_tokens
            
            # 3x3 for now 241030
            if tgt_height_idx == 0:
                h_dirs = [0,1,2]
            elif tgt_height_idx == h_num_tokens-1:
                h_dirs = [-2,-1,0]
            else:
                h_dirs = [-1,0,1]
            
            if tgt_width_idx == 0:
                w_dirs = [0,1,2]
            elif tgt_width_idx == w_num_tokens-1:
                w_dirs = [-2,-1,0]
            else:
                w_dirs = [-1,0,1]
            
            self.nn_visual_token_idxs_list = []
            for h_dir in h_dirs:
                for w_dir in w_dirs:
                    nn_visual_token_ids = (tgt_height_idx + h_dir) * h_num_tokens + (tgt_width_idx+w_dir)
                    # print(nn_visual_token_ids)
                    self.nn_visual_token_idxs_list.append(nn_visual_token_ids)
            assert len(self.nn_visual_token_idxs_list) == len(h_dirs) * len(w_dirs)
            image_features = image_features[:, 1:]
            
        elif self.select_feature == 'fix_sim_ablate_tgt_patch':
            assert self.pad_token_emb_path and self.pad_token_emb_flag, 'plz set pad_token_emb_flag to True'
            self.pad_visual_token = self.pad_visual_tokens.to(self.dtype).to(self.device)
            
            # get nn patch ids
            cls_features = image_features[:, :1].detach().clone()
            cls_token_norm = nn.functional.normalize(cls_features, p=2, dim=-1)
            
            patch_features = image_features[:, 1:].detach().clone()
            bs, L, D = patch_features.shape
            assert bs == 1, 'support bs=1 for now, 241030'
            patch_token_norm = nn.functional.normalize(patch_features, p=2, dim=-1)
            # cos_similarity = torch.bmm(cls_token_norm, patch_token_norm.transpose(1,2)).squeeze(1)
            cos_similarity = torch.matmul(patch_token_norm, cls_token_norm.transpose(1, 2)).squeeze(-1)  # [bs, L]
            # neg_cos_similarity = -cos_similarity
            # _, low_sim_index = torch.topk(neg_cos_similarity, k=int((L-1)*0.9), dim=-1, sorted=False)
            _, topk_cos_similarity_idx = torch.topk(
                cos_similarity, 
                k=int(self.patch_reduction_scale),
                dim=-1)  # [bs, L]
            
            tgt_visual_token_idxs = topk_cos_similarity_idx[0, -1].item()
            # print(f"center: {tgt_visual_token_idxs}")
            assert 0 <= tgt_visual_token_idxs < L
            # context level
            sqrt_L = math.sqrt(L)
            if not sqrt_L.is_integer():
                print(f"visual token number: {L} is invalid for math.sqrt()")
                raise ValueError
            self.nn_visual_token_idxs_list = [tgt_visual_token_idxs]
            image_features = image_features[:, 1:]
            
        elif self.select_feature == 'fix_id_ablate_tgt_patch':
            assert self.pad_token_emb_path and self.pad_token_emb_flag, 'plz set pad_token_emb_flag to True'
            self.pad_visual_token = self.pad_visual_tokens.to(self.dtype).to(self.device)
            
            patch_features = image_features[:, 1:].detach().clone()
            bs, L, D = patch_features.shape
            assert bs == 1, 'support bs=1 for now, 241030'
            
            tgt_visual_token_idxs = int(self.patch_reduction_scale)
            # print(f"center: {tgt_visual_token_idxs}")
            assert 0 <= tgt_visual_token_idxs < L
            # context level
            sqrt_L = math.sqrt(L)
            if not sqrt_L.is_integer():
                print(f"visual token number: {L} is invalid for math.sqrt()")
                raise ValueError
            self.nn_visual_token_idxs_list = [tgt_visual_token_idxs]
            image_features = image_features[:, 1:]
            
        else:
            raise ValueError(f"Unexpected select feature: {select_feature_type}")
        # print(image_features.shape)
        return image_features

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs)
            if isinstance(image_features, list):
                assert self.select_feature in ["fix_sim_context_pad_patch", "fix_id_context_pad_patch"]
                new_img_feats_ = []
                for q in image_features:
                    if q is not None:
                        new_img_feats_.append(q.to(images.dtype))
                    else:
                        new_img_feats_.append(q)
                image_features = new_img_feats_
            else:
                image_features = image_features.to(images.dtype)
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        _hidden_size = self.config.hidden_size
        if "slicefour" in self.select_feature:
            _hidden_size *= 4
        if "slice_m25811_f6" in self.select_feature:
            _hidden_size *= 5
        return _hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        _num_patches = (self.config.image_size // self.config.patch_size) ** 2
        if "cls_patch" in self.select_feature:
            _num_patches += 1
        return _num_patches

    @property
    def image_size(self):
        return self.config.image_size


class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):

        self.s2_scales = getattr(args, "s2_scales", "336,672,1008")
        self.s2_scales = list(map(int, self.s2_scales.split(",")))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        super().__init__(vision_tower, args, delay_load)

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, "unfreeze_mm_vision_tower", False):
            self.image_processor.size["shortest_edge"] = self.s2_image_size
            self.image_processor.crop_size["height"] = self.image_processor.crop_size["width"] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size["shortest_edge"] = self.s2_image_size
        self.image_processor.crop_size["height"] = self.image_processor.crop_size["width"] = self.s2_image_size

        self.is_loaded = True

    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size, split_forward=True)
                image_features.append(image_feature)
        else:
            image_features = multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size, split_forward=True)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
