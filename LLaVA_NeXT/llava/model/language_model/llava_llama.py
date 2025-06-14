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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig

from torch.nn import CrossEntropyLoss


# , LlamaModel, LlamaForCausalLM, GenerationConfig
# from .modeling_llama import LlamaModel, LlamaForCausalLM
from transformers import LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"
    temperature: float = 0.0  # reset to 0.0, previously 0.9 for Vicuna
    max_new_tokens: int = 1024
    do_sample: bool = False
    top_p: Optional[float] = None
    # rope_scaling: Optional[dict] = {}


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        LlamaForCausalLM.__init__(self, config)

        # configure default generation settings
        config.model_type = "llava_llama"
        # config.rope_scaling = None

        self.model = LlavaLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = None,
        cache_position=None,
        # pensieve and vcd
        images_cd: Optional[torch.FloatTensor] = None,
        cd_beta: Optional[torch.FloatTensor] = None,
        cd_alpha: Optional[torch.FloatTensor] = None,
        alpha_base: Optional[torch.FloatTensor] = None,
        alpha_noise: Optional[torch.FloatTensor] = None,
        alpha_nns: Optional[torch.FloatTensor] = None,
        images_racd: Optional[torch.FloatTensor] = None,
        inputs_embeds_cd: Optional[torch.FloatTensor] = None,
        inputs_embeds_racd_l: Optional[torch.FloatTensor] = None,
        racd_topk: Optional[int] = None,
        jsd_thres: Optional[float] = None,
        # dola
        early_exit_layers: Optional[List[int]] = None,
        mature_layer: Optional[int] = None,
        premature_layer: Optional[int] = None,
        candidate_premature_layers: Optional[List[int]] = None,
        relative_top: Optional[float] = None,
        #
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels
        # # dola decoding
        elif early_exit_layers is not None:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
            )
            logits_dict = {}
            # loss_dict = {}
            # print(len(outputs.hidden_states))
            for i, early_exit_layer in enumerate(early_exit_layers):
                # print(outputs.hidden_states[early_exit_layer][:, 0, 24:29])
                logits = self.lm_head(outputs.hidden_states[early_exit_layer])
                logits = logits.float()
                logits_dict[early_exit_layer] = logits
                # print(early_exit_layer)
                
            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                    # loss_dict[early_exit_layer] = loss
                    
            final_outputs = CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
            return logits_dict, final_outputs

        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        # pensieve
        images_cd: Optional[torch.Tensor] = None,
        image_cd_sizes: Optional[torch.Tensor] = None,
        images_racd: Optional[torch.Tensor] = None,
        image_racd_sizes: Optional[torch.Tensor] = None,
        cd_beta=None,
        cd_alpha=None,
        alpha_noise=None,
        alpha_nns=None,
        alpha_base=None,
        racd_topk=None,
        jsd_thres=None,
        #
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        modalities = kwargs.pop("modalities", None) if "modalities" in kwargs and modalities is None else modalities
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        origin_inputs = inputs.clone()
        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(
                inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
            
        if images_cd is not None:
            (_, _, _, _, inputs_embeds_cd, _) = self.prepare_inputs_labels_for_multimodal(
                origin_inputs, position_ids, attention_mask, None, None, images_cd, modalities, image_sizes=image_cd_sizes)
        
        if images_racd is not None:
            assert isinstance(images_racd, list) and len(images_racd) >= 1
            inputs_embeds_racd_l =[]
            for images_racd_, image_racd_sizes_ in zip(images_racd, image_racd_sizes):
                assert isinstance(images_racd_, list) and len(images_racd_) == 1
                assert isinstance(image_racd_sizes_, list) and len(image_racd_sizes_) == 1
                (_, _, _, _, inputs_embeds_racd_, _) = self.prepare_inputs_labels_for_multimodal(
                    origin_inputs, position_ids, attention_mask, None, None, images_racd_, modalities, image_sizes=image_racd_sizes_)
                inputs_embeds_racd_l.append(inputs_embeds_racd_)
            assert len(inputs_embeds_racd_l) == len(images_racd)
            
        # print(f"inputs_embeds.shape: {inputs_embeds.shape}")
        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, 
                                images_cd=images_cd if images_cd is not None else None,
                                inputs_embeds_cd=inputs_embeds_cd if images_cd is not None else None,
                                images_racd=images_racd if images_racd is not None else None,
                                inputs_embeds_racd_l=inputs_embeds_racd_l if images_racd is not None else None,
                                cd_beta=cd_beta if cd_beta is not None else None,
                                cd_alpha=cd_alpha if cd_alpha is not None else None,
                                alpha_noise=alpha_noise if alpha_noise is not None else None,
                                alpha_nns=alpha_nns if alpha_nns is not None else None,
                                alpha_base=alpha_base if alpha_base is not None else None,
                                racd_topk=racd_topk if racd_topk is not None else None,
                                jsd_thres=jsd_thres if jsd_thres is not None else None,
                                **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs
    
    def prepare_inputs_for_generation_cd(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images_cd", None)
        image_sizes = kwargs.pop("image_cd_sizes", None)
        inputs_embeds_cd = kwargs.get("inputs_embeds_cd")
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds_cd, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs

    def prepare_inputs_for_generation_racd(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images_racd", None)
        image_sizes = kwargs.pop("image_racd_sizes", None)
        inputs_embeds_racd_l = kwargs.get("inputs_embeds_racd_l")
        j = kwargs.get("nn_idx")
        inputs_embeds_racd = inputs_embeds_racd_l[j]
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds_racd, **kwargs)
        if images is not None:
            inputs["images"] = images[j]
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes[j]
        return inputs
    

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
