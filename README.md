
<div align="center">

<h2><center>[🔥ICLR 2026] LaTo: Landmark-tokenized Diffusion Transformer for Fine-grained Human Face Editing</h2>

Zhenghao Zhang∗, Ziying Zhang∗, Junchao Liao∗, Xiangyu Meng, Qiang Hu, Siyu Zhu, Xiaoyun Zhang, Long Qin, Weizhi Wang

\* equal contribution
<br>

<a href='https://arxiv.org/abs/2509.25731'><img src='https://img.shields.io/badge/ArXiv-2509.25731-red'></a>
<a href="https://github.com/alibaba/landmark-tokenized-dit"><img src='https://img.shields.io/badge/Github-Link-orange'></a>
<a href='https://modelscope.cn/models/Alibaba_Research_Intelligence_Computing/LaTo'><img src='https://img.shields.io/badge/🤖_ModelScope-weights-%23654dfc'></a>
<a href='https://huggingface.co/Alibaba-Research-Intelligence-Computing/LaTo'><img src='https://img.shields.io/badge/🤗_HuggingFace-weights-%23ff9e0e'></a>
</div>

This is the official repository for paper "LaTo: Landmark-tokenized Diffusion Transformer for Fine-grained Human Face Editing".

# 💡 Abstract

Recent multimodal models for instruction-based face editing enable semantic manipulation but still struggle with precise attribute control and identity preservation. Structural facial representations such as landmarks are effective for intermediate supervision, yet most existing methods treat them as rigid geometric constraints, which can degrade identity when conditional landmarks deviate significantly from the source (e.g., large expression or pose changes, inaccurate landmark estimates). To address these limitations, we propose LaTo, a landmark-tokenized diffusion transformer for fine-grained, identity-preserving face editing. Our key innovations include: (1) a landmark tokenizer that directly quantizes raw landmark coordinates into discrete facial tokens, obviating the need for dense pixel-wise correspondence; (2) a location-mapped positional encoding and a landmark-aware classifier-free guidance that jointly facilitate flexible yet decoupled interactions among instruction, geometry, and appearance, enabling strong identity preservation; and (3) a landmark predictor that leverages vision–language models to infer target landmarks from instructions and source images, whose structured chain-of-thought improves estimation accuracy and interactive control. To mitigate data scarcity, we curate HFL-150K, to our knowledge the largest benchmark for this task, containing over 150K real face pairs with fine-grained instructions. Extensive experiments show that LaTo outperforms state-of-the-art methods by 7.8% in identity preservation and 4.6% in semantic consistency. 

# 📑 Table of Contents

- [💡 Abstract](#-abstract)
- [🐍 Installation](#-installation)
- [📦 Model Weights](#-model-weights)
- [🔄 Inference](#-inference)
- [🤝 Acknowledgements](#-acknowledgements)
- [📄 Our previous work](#-our-previous-work)
- [📚 Citation](#-citation)

# 🐍 Installation
Dependencies: python (tested on python 3.10)

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# download the compatible flash-attn wheel from https://github.com/Dao-AILab/flash-attention/releases
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# if error when installing flash-attn, try install from source. See https://github.com/Dao-AILab/flash-attention
```

# 📦 Model Weights
```bash
bash scripts/download_model.sh
```

# 🔄 Inference
```bash
bash scripts/run_inference.sh
```

# 🤝 Acknowledgements

We would like to express our gratitude to the following open-source projects that have been instrumental in the development of our project: [Step1X-Edit](https://github.com/stepfun-ai/Step1X-Edit), [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)

Special thanks to the contributors of these libraries for their hard work and dedication!

# 📄 Our previous work

- [AnimateAnything: Fine Grained Open Domain Image Animation with Motion Guidance](https://github.com/alibaba/animate-anything)
- [Tora: Trajectory-oriented Diffusion Transformer for Video Generation](https://github.com/alibaba/tora)
- [Tora2: Motion and Appearance Customized Diffusion Transformer for Multi-Entity Video Generation](https://ali-videoai.github.io/Tora2_page/)


# 📚 Citation

```bibtex
@misc{zhang2026latolandmarktokenizeddiffusiontransformer,
      title={LaTo: Landmark-tokenized Diffusion Transformer for Fine-grained Human Face Editing}, 
      author={Zhenghao Zhang and Ziying Zhang and Junchao Liao and Xiangyu Meng and Qiang Hu and Siyu Zhu and Xiaoyun Zhang and Long Qin and Weizhi Wang},
      year={2026},
      eprint={2509.25731},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.25731}, 
}

```
