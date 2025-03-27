
<p align="center" width="100%">
<a target="_blank"><img src="assets/redundancy_analysis_pipeline.png" alt="Visual Redundancy Explanation through Language" style="width: 100%; min-width: 200px; display: block; margin: auto;"></a>
</p>

# Beyond Intermediate States: Explaining Visual Redundancy through Language

## 🙋🏼‍♂️🙋🏽🙋🏻‍♀️ Introduction

Multi-modal Large Langue Models (MLLMs) often process thousands of visual tokens, which consume a significant portion of the context window and impose a substantial computational burden. Prior work has empirically explored visual token pruning methods based on MLLMs’ intermediate states (e.g., attention scores). However, they have limitations in precisely defining visual redundancy due to their inability to capture the influence of visual tokens on MLLMs’ visual understanding (i.e., the predicted probabilities for textual token candidates). To address this issue, we manipulate the visual input and investigate variations in the textual output from both token-centric and context-centric perspectives, achieving intuitive and comprehensive analysis. Experimental results reveal that visual tokens with low ViT−[cls] association and low text-to-image attention scores can contain recognizable information and significantly contribute to images’ overall information. To develop a more reliable method for identifying and pruning redundant visual tokens, we integrate these two perspectives and introduce a context-independent condition to identify redundant prototypes from training images, which probes the redundancy of each visual token during inference. Extensive experiments on single-image, multi-image and video comprehension tasks demonstrate the effectiveness of our method, notably achieving 90% to 110% of the performance while pruning 80% to 90% of visual tokens.

## 🪜 Updates

- [2025-03-26]: 🎉🎉 Our [Paper](https://arxiv.org/abs/2503.20540) is available on Arxiv.

## 🖼️ Method Overview

<p align="center" width="100%">
<a target="_blank"><img src="assets/method.png" alt="Method Overview" style="width: 50%; min-width: 200px; display: block; margin: auto;"></a>
</p>

- We propose a novel method to identify and prune redundant visual tokens during MLLMs' inference stage.