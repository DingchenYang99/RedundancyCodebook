---
datasets:
- lmms-lab/LLaVA-OneVision-Data
language:
- en
- zh
library_name: transformers
license: apache-2.0
metrics:
- accuracy
tags:
- multimodal
model-index:
- name: llava-onevision-qwen-7b-ov
  results:
  - task:
      type: multimodal
    dataset:
      name: AI2D
      type: ai2d
    metrics:
    - type: accuracy
      value: 81.4
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: ChartQA
      type: chartqa
    metrics:
    - type: accuracy
      value: 80.0
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: DocVQA
      type: docvqa
    metrics:
    - type: accuracy
      value: 90.2
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: InfoVQA
      type: infovqa
    metrics:
    - type: accuracy
      value: 70.7
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: MathVerse
      type: mathverse
    metrics:
    - type: accuracy
      value: 26.2
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: MathVista
      type: mathvista
    metrics:
    - type: accuracy
      value: 63.2
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: MMBench
      type: mmbench
    metrics:
    - type: accuracy
      value: 80.8
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: MME-Perception
      type: mme-perception
    metrics:
    - type: score
      value: 1580
      name: score
      verified: true
  - task:
      type: multimodal
    dataset:
      name: MME-Cognition
      type: mme-cognition
    metrics:
    - type: score
      value: 418
      name: score
      verified: true
  - task:
      type: multimodal
    dataset:
      name: MMMU
      type: mmmu
    metrics:
    - type: accuracy
      value: 48.8
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: MMVet
      type: mmvet
    metrics:
    - type: accuracy
      value: 57.5
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: MMStar
      type: mmstar
    metrics:
    - type: accuracy
      value: 61.7
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: Seed-Bench
      type: seed-bench
    metrics:
    - type: accuracy
      value: 75.4
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: Science-QA
      type: science-qa
    metrics:
    - type: accuracy
      value: 96.0
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: ImageDC
      type: imagedc
    metrics:
    - type: accuracy
      value: 88.9
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: MMLBench
      type: mmlbench
    metrics:
    - type: accuracy
      value: 77.1
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: RealWorldQA
      type: realworldqa
    metrics:
    - type: accuracy
      value: 66.3
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: Vibe-Eval
      type: vibe-eval
    metrics:
    - type: accuracy
      value: 51.7
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: LLaVA-W
      type: llava-w
    metrics:
    - type: accuracy
      value: 90.7
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: LLaVA-Wilder
      type: l-wilder
    metrics:
    - type: accuracy
      value: 67.8
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: ActNet-QA
      type: actnet-qa
    metrics:
    - type: accuracy
      value: 56.6
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: EgoSchema
      type: egoschema
    metrics:
    - type: accuracy
      value: 60.1
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: MLVU
      type: mlvu
    metrics:
    - type: accuracy
      value: 64.7
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: MVBench
      type: mvbench
    metrics:
    - type: accuracy
      value: 56.7
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: NextQA
      type: nextqa
    metrics:
    - type: accuracy
      value: 79.4
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: PercepTest
      type: percepTest
    metrics:
    - type: accuracy
      value: 49.7
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: SeedBench
      type: seedbench
    metrics:
    - type: accuracy
      value: 56.9
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: VideoChatGPT
      type: videochatgpt
    metrics:
    - type: score
      value: 3.49
      name: score
      verified: true
  - task:
      type: multimodal
    dataset:
      name: VideoDC
      type: videodc
    metrics:
    - type: score
      value: 3.75
      name: score
      verified: true
  - task:
      type: multimodal
    dataset:
      name: VideoMME
      type: videomme
    metrics:
    - type: accuracy
      value: 58.2
      name: accuracy
      verified: true
---


# LLaVA-OneVision

![banner](https://i.postimg.cc/pL17YtG4/WX20240508-220230-2x.png)

Play with the model on the [LLaVA OneVision Chat](https://llava-onevision.lmms-lab.com/).

##  Table of Contents

1. [Model Summary](##model-summary)
2. [Use](##use)
3. [Limitations](##limitations)
4. [Training](##training)
5. [License](##license)
6. [Citation](##citation)

## Model Summary

The LLaVA-OneVision models are 0.5/7/72B parameter models trained on [LLaVA-OneVision](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data), based on Qwen2 language model with a context window of 32K tokens.

- **Repository:** [LLaVA-VL/LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT?tab=readme-ov-file)
- **Project Website:** [llava-onevision.lmms-lab.com](llava-onevision.lmms-lab.com)
- **Paper:** [LLaVA-OneVision](arxiv.org/abs/2408.03326)
- **Point of Contact:** [Bo Li](mailto:drluodian@gmail.com)
- **Languages:** English, Chinese


## Use

### Intended use

The model was trained on [LLaVA-OneVision Dataset](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data) and have the ability to interact with images, multi-image and videos. 

**Feel free to share your generations in the Community tab!**

### Generation

We provide the simple generation process for using our model. For more details, you could refer to [Github](https://github.com/LLaVA-VL/LLaVA-NeXT).

```python
# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch

import sys
import warnings

warnings.filterwarnings("ignore")
pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args

model.eval()

url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
question = DEFAULT_IMAGE_TOKEN + "\nWhat is shown in this image?"
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]


cont = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print(text_outputs)
```

# Training

## Model

- **Architecture:** SO400M + Qwen2
- **Pretraining Stage:** LCS-558K, 1 epoch, projector
- **Mid Stage:** A mixture of 4.7M high-quality synthetic data, 1 epoch, full model
- **Final-Image Stage:** A mixture of 3.6M single-image data, 1 epoch, full model
- **OneVision Stage:** A mixture of 1.6M single-image/multi-image/video data, 1 epoch, full model
- **Precision:** bfloat16

## Hardware & Software

- **GPUs:** 256 * Nvidia Tesla A100 (for whole model series training)
- **Orchestration:** [Huggingface Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
- **Neural networks:** [PyTorch](https://github.com/pytorch/pytorch)

# Citation
```
@article{li2024llavaonevision,
      title={LLaVA-OneVision}, 
}
```