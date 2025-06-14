#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import math
import re
import time
import torch
import torch.nn as nn
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print
import random


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        
        if not hasattr(self.config, 'add_faster_video'):
            if model_args.add_faster_video:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.faster_token = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
            rank0_print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_2dPool(self, image_feature, stride=2):
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, width = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature

    def encode_images(self, images):
        vision_tower = self.get_model().get_vision_tower()
        image_features = vision_tower(images)
        # image_features = self.get_model().get_vision_tower()(images)
        # image_features = self.get_model().vision_resampler(image_features, images=images)
        if vision_tower.select_feature in ['fix_sim_context_pad_patch', 'fix_id_context_pad_patch']:
            assert isinstance(image_features, list) and len(image_features) == 2
            image_features_prev, image_features_post = image_features
            image_features_prev = self.get_model().mm_projector(image_features_prev) \
                if image_features_prev is not None else None
            image_features_post = self.get_model().mm_projector(image_features_post) \
                if image_features_post is not None else None
                
            pad_visual_tokens = vision_tower.pad_visual_tokens.clone()
            
            if image_features_prev is None:
                pad_visual_tokens = pad_visual_tokens[None, ...].repeat(image_features_post.shape[0], 1, 1)
                image_features = torch.cat([pad_visual_tokens, image_features_post], dim=1)
            elif image_features_post is None:
                pad_visual_tokens = pad_visual_tokens[None, ...].repeat(image_features_prev.shape[0], 1, 1)
                image_features = torch.cat([image_features_prev, pad_visual_tokens], dim=1)
            else:
                assert image_features_prev.shape[0] == image_features_post.shape[0]
                pad_visual_tokens = pad_visual_tokens[None, ...].repeat(image_features_prev.shape[0], 1, 1)
                image_features = torch.cat([image_features_prev, pad_visual_tokens, image_features_post], dim=1)
        else:
            image_features = self.get_model().mm_projector(image_features)
        
        if vision_tower.select_feature in ['high_info_patch']:
            p_features = image_features.detach().clone()
            # print(patch_features.shape)
            bs, L, D = p_features.shape
            # assert bs <= 3, "support bs=1 only for now 241030, do not split sub-images."
            patch_token_norm = nn.functional.normalize(p_features, p=2, dim=-1)
            
            toxic_visual_tokens = vision_tower.toxic_visual_tokens
            # print(toxic_visual_tokens.shape)
            toxic_visual_tokens_norm = nn.functional.normalize(toxic_visual_tokens, p=2, dim=-1)
            toxic_visual_tokens_norm = toxic_visual_tokens_norm.unsqueeze(0).repeat(bs, 1, 1)
            # print(patch_token_norm.shape, toxic_visual_tokens_norm.shape)
            toxic_visual_tokens_norm = toxic_visual_tokens_norm.transpose(1, 2)
            # dist.broadcast(toxic_visual_tokens_norm, src=0)
            cos_similarity = torch.matmul(patch_token_norm, toxic_visual_tokens_norm)  # [bs, L, n]
            toxic_codebook_thres = vision_tower.toxic_codebook_thres  # .to(self.dtype).to(self.device)
            
            # method1 max per feat dim, work for llava-next
            max_cos_sim, max_cos_sim_idx = torch.max(cos_similarity, dim=-1)
            # # print(max_cos_sim)
            high_info_index_bin = max_cos_sim < toxic_codebook_thres  # [bs, L]
            # print(high_info_index_bin.shape)
            # === 
            
            select_patch_feat_list = [image_features[i][high_info_index_bin[i]] for i in range(bs)]
            
            if all(x.shape == select_patch_feat_list[0].shape for x in select_patch_feat_list):
                image_features = torch.stack(select_patch_feat_list)
            else:
                image_features = [q.unsqueeze(0) for q in select_patch_feat_list]
                
            vision_tower.patch_retain_mask_bin = high_info_index_bin
            vision_tower.num_visual_tokens_buffer += sum([y.shape[-2] for y in select_patch_feat_list])
            
        if vision_tower.select_feature in ['high_info_patch_fix_num']:
            p_features = image_features.detach().clone()
            # print(patch_features.shape)
            bs, L, D = p_features.shape
            assert bs == 1, "support bs=1 only for now 250304, do not split sub-images."
            patch_token_norm = nn.functional.normalize(p_features, p=2, dim=-1)
            
            toxic_visual_tokens = vision_tower.toxic_visual_tokens
            # print(toxic_visual_tokens.shape)
            toxic_visual_tokens_norm = nn.functional.normalize(toxic_visual_tokens, p=2, dim=-1)
            toxic_visual_tokens_norm = toxic_visual_tokens_norm.unsqueeze(0).repeat(bs, 1, 1)
            # print(patch_token_norm.shape, toxic_visual_tokens_norm.shape)
            toxic_visual_tokens_norm = toxic_visual_tokens_norm.transpose(1, 2)
            # dist.broadcast(toxic_visual_tokens_norm, src=0)
            cos_similarity = torch.matmul(patch_token_norm, toxic_visual_tokens_norm)  # [bs, L, n]
            toxic_codebook_thres = vision_tower.toxic_codebook_thres  # .to(self.dtype).to(self.device)
            
            # method1 max per feat dim, work for llava-next
            max_cos_sim, max_cos_sim_idx = torch.max(cos_similarity, dim=-1)
            # # print(max_cos_sim)
            
            sampled_L = round(L*toxic_codebook_thres)
            keep_indices = max_cos_sim.argsort(dim=-1, descending=False)[:, :sampled_L]
            keep_indices_sorted = keep_indices.sort(dim=-1).values
            keep_patch_features = image_features[torch.arange(bs).unsqueeze(1), keep_indices_sorted]
            
            high_info_index_bin = max_cos_sim < torch.topk(
                max_cos_sim, 
                k=round(L*(1-toxic_codebook_thres)), 
                dim=-1)[0][..., -1:]  # [bs, L]
            # print(high_info_index_bin.shape)
            
            # === 
            
            # select_patch_feat_list = [image_features[i][high_info_index_bin[i]] for i in range(bs)]
            
            # if all(x.shape == select_patch_feat_list[0].shape for x in select_patch_feat_list):
                # image_features = torch.stack(select_patch_feat_list)
            # else:
                # image_features = [q.unsqueeze(0) for q in select_patch_feat_list]
            image_features = keep_patch_features
            vision_tower.patch_retain_mask_bin = high_info_index_bin
            vision_tower.num_visual_tokens_buffer += image_features.shape[-2]
            
            
        elif vision_tower.select_feature in ['high_info_patch_flatten']:
            p_features = image_features.detach().clone()
            # print(patch_features.shape)
            bs, L, D = p_features.shape
            # assert bs <= 3, "support bs=1 only for now 241030, do not split sub-images."
            patch_token_norm = nn.functional.normalize(p_features, p=2, dim=-1)
            
            toxic_visual_tokens = vision_tower.toxic_visual_tokens
            # print(toxic_visual_tokens.shape)
            toxic_visual_tokens_norm = nn.functional.normalize(toxic_visual_tokens, p=2, dim=-1)
            toxic_visual_tokens_norm = toxic_visual_tokens_norm.unsqueeze(0).repeat(bs, 1, 1)
            # print(patch_token_norm.shape, toxic_visual_tokens_norm.shape)
            toxic_visual_tokens_norm = toxic_visual_tokens_norm.transpose(1, 2)
            # dist.broadcast(toxic_visual_tokens_norm, src=0)
            cos_similarity = torch.matmul(patch_token_norm, toxic_visual_tokens_norm)  # [bs, L, n]
            toxic_codebook_thres = vision_tower.toxic_codebook_thres  # .to(self.dtype).to(self.device)
            
            # method1 max per feat dim, work for llava-next
            max_cos_sim, max_cos_sim_idx = torch.max(cos_similarity, dim=-1)
            # # print(max_cos_sim)
            high_info_index_bin = max_cos_sim < toxic_codebook_thres  # [bs, L]
            # print(high_info_index_bin.shape)
            # === 
            
            select_patch_feat_list = [image_features[i][high_info_index_bin[i]] for i in range(bs)]
            
            # if all(x.shape == select_patch_feat_list[0].shape for x in select_patch_feat_list):
                # image_features = torch.stack(select_patch_feat_list)
            # else:
            if bs == 1:
                image_features = [q.unsqueeze(0) for q in select_patch_feat_list]
            elif bs > 1:  # split sub images, each sub image with different L
                image_features = [torch.cat([q.unsqueeze(0) for q in select_patch_feat_list], dim=-2)]
                    
        elif vision_tower.select_feature in ['high_info_patch_double_smx']:
            p_features = image_features.detach().clone()
            # print(patch_features.shape)
            bs, L, D = p_features.shape
            # assert bs <= 3, "support bs=1 only for now 241030, do not split sub-images."
            patch_token_norm = nn.functional.normalize(p_features, p=2, dim=-1)
            
            toxic_visual_tokens = vision_tower.toxic_visual_tokens
            # print(toxic_visual_tokens.shape)
            toxic_visual_tokens_norm = nn.functional.normalize(toxic_visual_tokens, p=2, dim=-1)
            toxic_visual_tokens_norm = toxic_visual_tokens_norm.unsqueeze(0).repeat(bs, 1, 1)
            # print(patch_token_norm.shape, toxic_visual_tokens_norm.shape)
            toxic_visual_tokens_norm = toxic_visual_tokens_norm.transpose(1, 2)
            # dist.broadcast(toxic_visual_tokens_norm, src=0)
            cos_similarity = torch.matmul(patch_token_norm, toxic_visual_tokens_norm)  # [bs, L, n]
            toxic_codebook_thres = vision_tower.toxic_codebook_thres  # .to(self.dtype).to(self.device)
            
            # method1 max per feat dim, work for llava-next
            # max_cos_sim, max_cos_sim_idx = torch.max(cos_similarity, dim=-1)
            # # # print(max_cos_sim)
            # high_info_index_bin = max_cos_sim < toxic_codebook_thres  # [bs, L]
            # # print(high_info_index_bin.shape)
            # === 
            
            # method2 double softmax, work for llava-ov, but do not work for llava-next
            assert len(cos_similarity.shape) == 3
            cos_similarity_smx_dim1 = nn.functional.softmax(cos_similarity, dim=1)
            cos_similarity_smx_dim2 = nn.functional.softmax(cos_similarity, dim=-1)
            cos_similarity_smx = cos_similarity_smx_dim1 * cos_similarity_smx_dim2  # [bs, L, n]
            max_cos_smx, max_cos_smx_idx = torch.max(cos_similarity_smx, dim=-1)  # [bs, L]
            # print(max_cos_smx)
            # max_cos_smx_log = -torch.log(max_cos_smx)
            # print(max_cos_smx_log)
            # high_info_index_bin = max_cos_smx < toxic_codebook_thres  # [bs, L]
            high_info_index_bin = max_cos_smx < torch.topk(
                max_cos_smx, 
                k=round(L*(1-toxic_codebook_thres)), 
                dim=-1)[0][..., -1:]  # [bs, L]
            # === 
            
            select_patch_feat_list = [image_features[i][high_info_index_bin[i]] for i in range(bs)]
            
            if all(x.shape == select_patch_feat_list[0].shape for x in select_patch_feat_list):
                image_features = torch.stack(select_patch_feat_list)
            else:
                image_features = [q.unsqueeze(0) for q in select_patch_feat_list]
                
        elif vision_tower.select_feature in ['high_info_patch_single_smx']:
            p_features = image_features.detach().clone()
            # print(patch_features.shape)
            bs, L, D = p_features.shape
            # assert bs <= 3, "support bs=1 only for now 241030, do not split sub-images."
            patch_token_norm = nn.functional.normalize(p_features, p=2, dim=-1)
            
            toxic_visual_tokens = vision_tower.toxic_visual_tokens
            # print(toxic_visual_tokens.shape)
            toxic_visual_tokens_norm = nn.functional.normalize(toxic_visual_tokens, p=2, dim=-1)
            toxic_visual_tokens_norm = toxic_visual_tokens_norm.unsqueeze(0).repeat(bs, 1, 1)
            # print(patch_token_norm.shape, toxic_visual_tokens_norm.shape)
            toxic_visual_tokens_norm = toxic_visual_tokens_norm.transpose(1, 2)
            # dist.broadcast(toxic_visual_tokens_norm, src=0)
            cos_similarity = torch.matmul(patch_token_norm, toxic_visual_tokens_norm)  # [bs, L, n]
            toxic_codebook_thres = vision_tower.toxic_codebook_thres  # .to(self.dtype).to(self.device)
            
            # method1 max per feat dim, work for llava-next
            # max_cos_sim, max_cos_sim_idx = torch.max(cos_similarity, dim=-1)
            # # # print(max_cos_sim)
            # high_info_index_bin = max_cos_sim < toxic_codebook_thres  # [bs, L]
            # # print(high_info_index_bin.shape)
            # === 
            
            # method2 double softmax, work for llava-ov, but do not work for llava-next
            assert len(cos_similarity.shape) == 3
            # cos_similarity_smx_dim1 = nn.functional.softmax(cos_similarity, dim=1)
            cos_similarity_smx_dim2 = nn.functional.softmax(cos_similarity, dim=-1)
            # cos_similarity_smx = cos_similarity_smx_dim1 * cos_similarity_smx_dim2  # [bs, L, n]
            max_cos_smx, max_cos_smx_idx = torch.max(cos_similarity_smx_dim2, dim=-1)  # [bs, L]
            # print(max_cos_smx)
            # max_cos_smx_log = -torch.log(max_cos_smx)
            # print(max_cos_smx_log)
            # high_info_index_bin = max_cos_smx < toxic_codebook_thres  # [bs, L]
            high_info_index_bin = max_cos_smx < torch.topk(
                max_cos_smx, 
                k=round(L*(1-toxic_codebook_thres)), 
                dim=-1)[0][..., -1:]  # [bs, L]
            # === 
            
            select_patch_feat_list = [image_features[i][high_info_index_bin[i]] for i in range(bs)]
            
            if all(x.shape == select_patch_feat_list[0].shape for x in select_patch_feat_list):
                image_features = torch.stack(select_patch_feat_list)
            else:
                image_features = [q.unsqueeze(0) for q in select_patch_feat_list]
                
            # vision_tower.num_visual_tokens_buffer += sum([y.shape[-2] for y in select_patch_feat_list])
            
        elif vision_tower.select_feature in ['fix_sim_patch_pad']:
            raise NotImplementedError
            # pad to 576 length
            # pad_visual_tokens = vision_tower.pad_visual_tokens
            # assert len(pad_visual_tokens.shape) == 2
            # assert pad_visual_tokens.shape[0] == 1
            # patch_features_ = image_features.detach().clone()
            # bs, L, D = patch_features_.shape
            # assert bs == 1, "support bs=1 only for now for padding img tokens, do not split sub-images."
            # padding_tensor = pad_visual_tokens.expand(L, D)
            # padded_visual_features = torch.zeros_like(patch_features_, dtype=patch_features_.dtype, 
            #                                           device=patch_features_.device)
            # retain_idxs = vision_tower.retain_visual_token_idxs_in_pad
            # for i in range(bs):
            #     padded_visual_features[i] = padding_tensor.clone()
            #     for retain_idx in retain_idxs:
            #         padded_visual_features[i, retain_idx] = image_features[i, retain_idx]
            
            # image_features = padded_visual_features
            
            # pad to config.min_single_token_dup_num
            pad_visual_tokens = vision_tower.pad_visual_tokens.clone()
            dup_times = vision_tower.min_single_token_dup_num - image_features.shape[1]
            pad_visual_tokens = pad_visual_tokens[None, ...].repeat(
                image_features.shape[0], dup_times, 1)
            image_features = torch.cat([image_features, pad_visual_tokens], dim=1)
        
        elif vision_tower.select_feature in ['fix_sim_ablate_context_tgt_patch',
                                             'fix_sim_ablate_tgt_patch',
                                             'fix_id_ablate_tgt_patch',
                                             'fix_id_ablate_context_tgt_patch']:
            pad_visual_tokens = vision_tower.pad_visual_tokens.clone()
            assert len(pad_visual_tokens.shape) == 2
            assert pad_visual_tokens.shape[0] == 1
            nn_visual_token_idxs_list = vision_tower.nn_visual_token_idxs_list
            
            bs, L, D = image_features.shape
            for i in range(bs):
                for l in nn_visual_token_idxs_list:
                    image_features[i, l] = pad_visual_tokens[0].clone()
            
        # print(image_features.shape)
        return image_features
    
    def encode_multimodals(self, videos_or_images, video_idx_in_batch, split_sizes=None):
        videos_or_images_features = self.get_model().get_vision_tower()(videos_or_images)
        per_videos_or_images_features = torch.split(videos_or_images_features, split_sizes, dim=0)  # tuple, (dim_1, 576, 4096)
        all_videos_or_images_features = []
        all_faster_video_features = []
        cur_mm_spatial_pool_stride = self.config.mm_spatial_pool_stride

        for idx, feat in enumerate(per_videos_or_images_features):
            
            feat = self.get_model().mm_projector(feat)
            faster_video_feature = 0
            slower_img_feat = 0
            if idx in video_idx_in_batch and cur_mm_spatial_pool_stride > 1:
                slower_img_feat = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
                if self.config.add_faster_video:
                    cur_mm_spatial_pool_stride = cur_mm_spatial_pool_stride * 2
                    faster_video_feature = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
            if slower_img_feat != 0:
                all_videos_or_images_features.append(slower_img_feat)
            else:
                all_videos_or_images_features.append(feat)
            all_faster_video_features.append(faster_video_feature)
        return all_videos_or_images_features,all_faster_video_features

    def add_token_per_grid(self, image_feature):
        resize_h = int(math.sqrt(image_feature.shape[1]))
        num_frames = image_feature.shape[0]
        feature_dim = image_feature.shape[-1]

        image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        if getattr(self.config, "add_faster_video", False):
            # import pdb; pdb.set_trace()
            # (3584, 832, 14) -> (3584, 64, 13, 14)
            image_feature = image_feature.view(feature_dim, num_frames,resize_h, -1)
            #  (3584, 64, 13, 14) -> (64, 13, 14, 3584)
            image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
            # (64, 13, 14, 3584) -> (64, 13*14, 3584)
            image_feature = image_feature.flatten(1, 2)
            # import pdb; pdb.set_trace()
            return image_feature
        # import pdb; pdb.set_trace()
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        return image_feature

    def add_token_per_frame(self, image_feature):
        image_feature = image_feature.permute(2, 0, 1).contiguous()
        image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        image_feature = image_feature.permute(1, 2, 0).contiguous()
        return image_feature

    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None):
        vision_tower = self.get_vision_tower()
        # rank_print(modalities)
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if isinstance(modalities, str):
            modalities = [modalities]

        # import pdb; pdb.set_trace()
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            encoded_image_features = self.encode_images(concat_images)
            # image_features,all_faster_video_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)

            # This is a list, each element is [num_images, patch * patch, dim]
            # rank_print(f"Concat images : {concat_images.shape}")
            if isinstance(encoded_image_features, list) or isinstance(encoded_image_features, tuple):
                pass
            else:
                encoded_image_features = torch.split(encoded_image_features, split_sizes)
            image_features = []
            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
                    if vision_tower.select_feature == 'patch':
                        image_features.append(self.get_2dPool(image_feat))
                    elif vision_tower.select_feature == 'video_random_patch':
                        random_patch_num = vision_tower.random_patch_num
                        patch_features = self.get_2dPool(image_feat)
                        bs, L, D = patch_features.shape
                        assert 0 < random_patch_num < L, f'plz assign random_patch_num within (0,{L}) for llava-ov'
                        keep_indices = torch.rand(bs, L).argsort(dim=1)[:, :random_patch_num]
                        keep_indices_sorted = keep_indices.sort(dim=1).values
                        keep_patch_features = patch_features[torch.arange(bs).unsqueeze(1), keep_indices_sorted]
                        image_features.append(keep_patch_features)
                        vision_tower.num_visual_tokens_buffer = sum([x.shape[-2] for x in keep_patch_features])
                    elif vision_tower.select_feature == 'video_high_info_patch':
                        patch_features = self.get_2dPool(image_feat)
                        bs, L, D = patch_features.shape
                        patch_token_norm = nn.functional.normalize(patch_features, p=2, dim=-1)
                        toxic_visual_tokens = vision_tower.toxic_visual_tokens
                        toxic_visual_tokens_norm = nn.functional.normalize(toxic_visual_tokens, p=2, dim=-1)
                        toxic_visual_tokens_norm = toxic_visual_tokens_norm.unsqueeze(0).repeat(bs, 1, 1)
                        toxic_visual_tokens_norm = toxic_visual_tokens_norm.transpose(1, 2)
                        cos_similarity = torch.matmul(patch_token_norm, toxic_visual_tokens_norm)  # [bs, L, n]
                        toxic_codebook_thres = vision_tower.toxic_codebook_thres  # .to(self.dtype).to(self.device)
                        
                        # method1 adaptive number for each image
                        max_cos_sim, max_cos_sim_idx = torch.max(cos_similarity, dim=-1)
                        high_info_index_bin = max_cos_sim < toxic_codebook_thres  # [bs, L]
                        # === 
                        
                        keep_patch_feat_list = [patch_features[i][high_info_index_bin[i]] for i in range(bs)]
                        keep_patch_feat = torch.cat(keep_patch_feat_list, dim=0)
                        image_features.append(keep_patch_feat)
                        vision_tower.num_visual_tokens_buffer = keep_patch_feat.shape[0]
                    elif vision_tower.select_feature == 'video_high_info_patch_double_smx':
                        patch_features = self.get_2dPool(image_feat)
                        bs, L, D = patch_features.shape
                        patch_token_norm = nn.functional.normalize(patch_features, p=2, dim=-1)
                        toxic_visual_tokens = vision_tower.toxic_visual_tokens
                        toxic_visual_tokens_norm = nn.functional.normalize(toxic_visual_tokens, p=2, dim=-1)
                        toxic_visual_tokens_norm = toxic_visual_tokens_norm.unsqueeze(0).repeat(bs, 1, 1)
                        toxic_visual_tokens_norm = toxic_visual_tokens_norm.transpose(1, 2)
                        cos_similarity = torch.matmul(patch_token_norm, toxic_visual_tokens_norm)  # [bs, L, n]
                        toxic_codebook_thres = vision_tower.toxic_codebook_thres  # .to(self.dtype).to(self.device)
                        
                        # method2 double softmax, work for llava-ov, but do not work for llava-next
                        assert cos_similarity.ndim == 3
                        cos_similarity_smx_dim1 = nn.functional.softmax(cos_similarity, dim=1)
                        cos_similarity_smx_dim2 = nn.functional.softmax(cos_similarity, dim=-1)
                        cos_similarity_smx = cos_similarity_smx_dim1 * cos_similarity_smx_dim2  # [bs, L, n]
                        max_cos_smx, max_cos_smx_idx = torch.max(cos_similarity_smx, dim=-1)  # [bs, L]
                        # max_cos_smx_log = -torch.log(max_cos_smx)
                        # high_info_index_bin = max_cos_smx < toxic_codebook_thres  # [bs, L]
                        high_info_index_bin = max_cos_smx < torch.topk(
                            max_cos_smx, 
                            k=round(L*(1-toxic_codebook_thres)), 
                            dim=-1)[0][..., -1:]  # [bs, L]
                        # === 
                        
                        keep_patch_feat_list = [patch_features[i][high_info_index_bin[i]] for i in range(bs)]
                        keep_patch_feat = torch.cat(keep_patch_feat_list, dim=0)
                        image_features.append(keep_patch_feat)
                        vision_tower.num_visual_tokens_buffer = keep_patch_feat.shape[0]
                    else:
                        print(f"unknown select_feature method {vision_tower.select_feature}")
                        raise NotImplementedError
                else:
                    image_features.append(image_feat)
            # image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
            # rank_print(f"Encoded image feats : {[x.shape for x in image_features]}")
            # image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")

            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]

            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    # rank0_print("At least we are reaching here")
                    # import pdb; pdb.set_trace()
                    if image_idx in video_idx_in_batch:  # video operations
                        # rank0_print("Video")
                        if mm_newline_position == "grid":
                            # Grid-wise
                            image_feature = self.add_token_per_grid(image_feature)
                            if getattr(self.config, "add_faster_video", False):
                                faster_video_feature = self.add_token_per_grid(all_faster_video_features[image_idx])
                                # Add a token for each frame
                                concat_slow_fater_token = []
                                # import pdb; pdb.set_trace()
                                for _ in range(image_feature.shape[0]):
                                    if _ % self.config.faster_token_stride == 0:
                                        concat_slow_fater_token.append(torch.cat((image_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                    else:
                                        concat_slow_fater_token.append(torch.cat((faster_video_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                # import pdb; pdb.set_trace()
                                image_feature = torch.cat(concat_slow_fater_token)

                                # print("!!!!!!!!!!!!")
                        
                            new_image_features.append(image_feature)
                        elif mm_newline_position == "frame":
                            # Frame-wise
                            image_feature = self.add_token_per_frame(image_feature)

                            new_image_features.append(image_feature.flatten(0, 1))
                            
                        elif mm_newline_position == "one_token":
                            # one-token
                            if image_feature.ndim == 3:
                                image_feature = image_feature.flatten(0, 1)
                            elif image_feature.ndim == 2:
                                pass
                            else:
                                raise ValueError
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                            new_image_features.append(image_feature)      
                        elif mm_newline_position == "no_token":
                            new_image_features.append(image_feature.flatten(0, 1))
                        else:
                            raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                    elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                        # rank0_print("Single-images")
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1))

                        if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                            except Exception as e:
                                rank0_print(f"Error: {e}")
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                            unit = image_feature.shape[2]
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        new_image_features.append(image_feature)
                    else:  # single image operations
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)
                        # print(image_feature.shape)
                        new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)
        # print(f"image_features.shape: {image_features.shape}")
        # print(len(image_features), image_features[0].shape)
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        # rank_print(f"Total images : {len(image_features)}")

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images)
            # print(batch_idx)
            # print(num_images)
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    # print(f"cur_image_idx: {cur_image_idx}, cur_image_features.shape: {cur_image_features.shape}")
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # import pdb; pdb.set_trace()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        # TODO: Hard code for control loss spike
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        # rank0_print("Prepare pos id")

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # rank0_print("tokenizer padding")

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        # import pdb; pdb.set_trace()
        # rank0_print("Finish preparing")
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
