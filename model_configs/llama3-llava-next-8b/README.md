---
{}
---

# LLaVA Model Card

## Model Details

Model type: LLaVA is an open-source chatbot trained by fine-tuning LLM on multimodal instruction-following data. It is an auto-regressive language model, based on the transformer architecture. 

Base LLM: meta-llama/Meta-Llama-3-8B-Instruct

### Model Description

**Repository:** https://github.com/LLaVA-VL/LLaVA-NeXT

**Primary intended uses:** The primary use of LLaVA is research on large multimodal models and chatbots. This is only for research exploration, and prohibited for commercial usage.

**Primary intended users:** The primary intended users of the model are researchers and hobbyists in computer vision, natural language processing, machine learning, and artificial intelligence.

### License Notices
  
This project utilizes certain datasets and checkpoints that are subject to their respective original licenses. Users must comply with all terms and conditions of these original licenses, including but not limited to the OpenAI Terms of Use for the dataset and the specific licenses for base language models for checkpoints trained using the dataset (e.g. Llama-1/2 community license for LLaMA-2 and Vicuna-v1.5, [Tongyi Qianwen LICENSE AGREEMENT](https://github.com/QwenLM/Qwen/blob/main/Tongyi%20Qianwen%20LICENSE%20AGREEMENT) and [META LLAMA 3 COMMUNITY LICENSE AGREEMENT](https://llama.meta.com/llama3/license/)). This project does not impose any additional constraints beyond those stipulated in the original licenses. Furthermore, users are reminded to ensure that their use of the dataset and checkpoints is in compliance with all applicable laws and regulations.

## How to Get Started with the Model

Use the code below to get started with the model.

[More Information Needed]

## Training Details

### Training Procedure

We conducted the training on LLaVA-1.6's codebase with adding support of Llama-3 and Qwen model.

### Training Hyperparameters

```shell
LLM_VERSION="meta-llama/Meta-Llama-3-8B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="openai/clip-vit-large-patch14-336"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

PROMPT_VERSION=plain
PRETRAIN_DATA_VERSION="blip558k"
############### Pretrain ################

BASE_RUN_NAME="llavanext-${LLM_VERSION_CLEAN}-${VISION_MODEL_VERSION_CLEAN}-pretrain_${PRETRAIN_DATA_VERSION}_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

PROMPT_VERSION="llava_llama_3"
MID_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-blip558k_pretrain_plain_la_1_6mix_ft"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

torchrun # with necessary torchrun information for distributed training\
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $LLM_VERSION \
    --version $PROMPT_VERSION \
    --data_path="/path/to/data/llava_instruct/llava1_6mix.json" \
    --image_folder /path/to/data/llava_data \
    --pretrain_mm_mlp_adapter="./checkpoints/projectors/${BASE_RUN_NAME}/mm_projector.bin" \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir "./checkpoints/${MID_RUN_NAME}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True

```

### Training Data

- 558K filtered image-text pairs from LAION/CC/SBU, captioned by BLIP.
- 158K GPT-generated multimodal instruction-following data.
- 500K academic-task-oriented VQA data mixture.
- 50K GPT-4V data mixture.
- 40K ShareGPT data.

#### Speeds, Sizes, Times [optional]

The training cost is ~15-20 hours on 2 x 8 NVIDIA A100-SXM4-80GB (may vary due to hardware differences). 

[More Information Needed]

## Evaluation

The evaluation is conducted with the support of [`lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval)