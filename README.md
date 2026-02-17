# 🔤 Awesome LLM Tokenization [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

<p align="center">
  <img src="https://img.shields.io/badge/Papers-120%2B-blue" alt="Papers">
  <img src="https://img.shields.io/badge/Tools-30%2B-green" alt="Tools">
  <img src="https://img.shields.io/badge/Last%20Updated-February%202026-orange" alt="Last Updated">
  <img src="https://img.shields.io/badge/PRs-Welcome-brightgreen" alt="PRs Welcome">
</p>

<p align="center">
  <em>A curated collection of research papers, tools, libraries, and resources on <strong>tokenization for Large Language Models</strong>.</em>
</p>

---

**Tokenization** is the foundational preprocessing step in every LLM pipeline, yet it remains one of the most underappreciated areas of research. The choice of tokenizer directly impacts model performance, multilingual capability, computational efficiency, arithmetic reasoning, code generation, and even hallucination patterns. This repository systematically organizes the research landscape around LLM tokenization — from classical subword algorithms to emerging tokenizer-free architectures.

> 📬 **Contributions welcome!** If you know a paper, tool, or resource that should be here, please open an issue or submit a pull request. See [Contributing](#contributing).

---

## Contents

- [Surveys & Overviews](#surveys--overviews)
- [Core Tokenization Algorithms](#core-tokenization-algorithms)
  - [Byte Pair Encoding (BPE)](#byte-pair-encoding-bpe)
  - [WordPiece](#wordpiece)
  - [Unigram Language Model](#unigram-language-model)
  - [SentencePiece](#sentencepiece)
- [Vocabulary Construction & Optimization](#vocabulary-construction--optimization)
- [Byte-Level & Character-Level Models](#byte-level--character-level-models)
- [Tokenizer-Free Architectures](#tokenizer-free-architectures)
- [Multilingual & Cross-Lingual Tokenization](#multilingual--cross-lingual-tokenization)
- [Tokenization for Code & Structured Data](#tokenization-for-code--structured-data)
- [Tokenization & Arithmetic / Numerical Reasoning](#tokenization--arithmetic--numerical-reasoning)
- [Multimodal Tokenization](#multimodal-tokenization)
- [Tokenizer Efficiency & Compression](#tokenizer-efficiency--compression)
- [Tokenization Analysis & Evaluation](#tokenization-analysis--evaluation)
- [Positional Encoding & Token Representation](#positional-encoding--token-representation)
- [Detokenization & Token Healing](#detokenization--token-healing)
- [Tools & Libraries](#tools--libraries)
- [Tokenizers of Notable LLMs](#tokenizers-of-notable-llms)
- [Blog Posts & Tutorials](#blog-posts--tutorials)
- [Talks & Videos](#talks--videos)
- [Datasets & Benchmarks](#datasets--benchmarks)
- [Contributing](#contributing)

---

## Surveys & Overviews

| Paper | Venue | Year | Links |
|-------|-------|------|-------|
| **Tokenization in the Theory of Knowledge** | arXiv | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2404.18261-b31b1b.svg)](https://arxiv.org/abs/2404.18261) |
| **A Survey on Subword Tokenization for NLP** | IEEE Access | 2023 | [![arXiv](https://img.shields.io/badge/arXiv-2311.02312-b31b1b.svg)](https://arxiv.org/abs/2311.02312) |
| **Tokenization Matters! Degrading Large Language Models through Challenging Their Tokenization** | arXiv | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2405.17067-b31b1b.svg)](https://arxiv.org/abs/2405.17067) |
| **Getting the Most Out of Your Tokenizer for Pre-Training and Domain Adaptation** | ACL | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2402.01035-b31b1b.svg)](https://arxiv.org/abs/2402.01035) |
| **On the Effect of Tokenization on Downstream Performance of Language Models** | arXiv | 2023 | [![arXiv](https://img.shields.io/badge/arXiv-2304.10774-b31b1b.svg)](https://arxiv.org/abs/2304.10774) |
| **Tokenizer Choice For LLM Training: Negligible or Crucial?** | NeurIPS Workshop | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2310.08754-b31b1b.svg)](https://arxiv.org/abs/2310.08754) |

---

## Core Tokenization Algorithms

### Byte Pair Encoding (BPE)

BPE is the most widely used tokenization algorithm in modern LLMs (GPT, LLaMA, Mistral, etc.). Originally a data compression technique, it was adapted for NLP by iteratively merging the most frequent pair of tokens.

| Paper | Venue | Year | Links |
|-------|-------|------|-------|
| **A New Algorithm for Data Compression** *(Original BPE)* | C Users Journal | 1994 | [Paper](https://www.derczynski.com/papers/archive/BPE_Gage.pdf) |
| **Neural Machine Translation of Rare Words with Subword Units** *(BPE for NLP)* | ACL | 2016 | [![arXiv](https://img.shields.io/badge/arXiv-1508.07909-b31b1b.svg)](https://arxiv.org/abs/1508.07909) [![GitHub](https://img.shields.io/badge/GitHub-Code-black)](https://github.com/rsennrich/subword-nmt) |
| **BPE-Dropout: Simple and Effective Subword Regularization** | ACL | 2020 | [![arXiv](https://img.shields.io/badge/arXiv-1910.13267-b31b1b.svg)](https://arxiv.org/abs/1910.13267) |
| **Language Model Is All You Need: A Fast and Effective Byte Pair Encoding Training Algorithm** | arXiv | 2023 | [![arXiv](https://img.shields.io/badge/arXiv-2306.16837-b31b1b.svg)](https://arxiv.org/abs/2306.16837) |
| **Byte Pair Encoding is Suboptimal for Language Model Pretraining** | Findings of ACL | 2020 | [![arXiv](https://img.shields.io/badge/arXiv-2004.03720-b31b1b.svg)](https://arxiv.org/abs/2004.03720) |
| **BBPE: Byte-Level Byte-Pair Encoding** *(used in GPT-2/3/4)* | — | 2019 | [Blog](https://openai.com/research/better-language-models) |
| **Efficient BPE Training with Suffix Arrays** | arXiv | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2407.07413-b31b1b.svg)](https://arxiv.org/abs/2407.07413) |
| **minbpe: Minimal BPE Implementation** | — | 2024 | [![GitHub](https://img.shields.io/badge/GitHub-Code-black)](https://github.com/karpathy/minbpe) |

### WordPiece

WordPiece is the tokenization algorithm behind BERT and its derivatives. Unlike BPE, it uses a likelihood-based criterion for merge selection.

| Paper | Venue | Year | Links |
|-------|-------|------|-------|
| **Japanese and Korean Voice Search** *(Original WordPiece)* | IEEE ICASSP | 2012 | [Paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf) |
| **Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation** *(WordPiece for NMT)* | arXiv | 2016 | [![arXiv](https://img.shields.io/badge/arXiv-1609.08144-b31b1b.svg)](https://arxiv.org/abs/1609.08144) |
| **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** | NAACL | 2019 | [![arXiv](https://img.shields.io/badge/arXiv-1810.04805-b31b1b.svg)](https://arxiv.org/abs/1810.04805) |
| **Fast WordPiece Tokenization** | EMNLP | 2021 | [![arXiv](https://img.shields.io/badge/arXiv-2012.15524-b31b1b.svg)](https://arxiv.org/abs/2012.15524) |

### Unigram Language Model

The Unigram model takes a probabilistic approach: starting from a large vocabulary, it iteratively removes tokens that least reduce the overall likelihood.

| Paper | Venue | Year | Links |
|-------|-------|------|-------|
| **Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates** | ACL | 2018 | [![arXiv](https://img.shields.io/badge/arXiv-1804.10959-b31b1b.svg)](https://arxiv.org/abs/1804.10959) |
| **Applying the Optimal Encoding Framework to Subword Tokenization** | arXiv | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2402.14576-b31b1b.svg)](https://arxiv.org/abs/2402.14576) |

### SentencePiece

SentencePiece is a language-independent tokenizer that implements both BPE and Unigram algorithms, treating the input as raw Unicode. It is used by T5, LLaMA, Gemma, and many multilingual models.

| Paper | Venue | Year | Links |
|-------|-------|------|-------|
| **SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing** | EMNLP (Demo) | 2018 | [![arXiv](https://img.shields.io/badge/arXiv-1808.06226-b31b1b.svg)](https://arxiv.org/abs/1808.06226) [![GitHub](https://img.shields.io/badge/GitHub-Code-black)](https://github.com/google/sentencepiece) |

---

## Vocabulary Construction & Optimization

How vocabularies are built, sized, and optimized has a major impact on downstream model performance, especially for multilingual and domain-specific applications.

| Paper | Venue | Year | Links |
|-------|-------|------|-------|
| **Optimal Subword Tokenization for Neural Machine Translation** | Findings of ACL | 2023 | [![arXiv](https://img.shields.io/badge/arXiv-2305.12331-b31b1b.svg)](https://arxiv.org/abs/2305.12331) |
| **Do All Languages Cost the Same? Tokenization in the Era of Commercial Language Models** | EMNLP | 2023 | [![arXiv](https://img.shields.io/badge/arXiv-2305.13707-b31b1b.svg)](https://arxiv.org/abs/2305.13707) |
| **An Optimal Transport Approach to Vocabulary Adaptation for Domain-Specific LLMs** | arXiv | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2405.09364-b31b1b.svg)](https://arxiv.org/abs/2405.09364) |
| **Vocabulary Adaptation for Continual Pre-training of Language Models** | ICLR | 2025 | [![arXiv](https://img.shields.io/badge/arXiv-2401.09359-b31b1b.svg)](https://arxiv.org/abs/2401.09359) |
| **How to Grow a (Product) Tree: Building a Hierarchical Tokenizer** | KDD | 2023 | [![arXiv](https://img.shields.io/badge/arXiv-2305.03038-b31b1b.svg)](https://arxiv.org/abs/2305.03038) |
| **Predicting the Quality of Subword Tokenization** | Findings of EMNLP | 2023 | [![arXiv](https://img.shields.io/badge/arXiv-2305.09167-b31b1b.svg)](https://arxiv.org/abs/2305.09167) |
| **Vocabulary Size Matters: Scaling Laws for LLM Tokenizers** | arXiv | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2407.13623-b31b1b.svg)](https://arxiv.org/abs/2407.13623) |
| **Obeying the Tokenizer: Constrained Generation for Language Models** | arXiv | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2404.04767-b31b1b.svg)](https://arxiv.org/abs/2404.04767) |
| **How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models** | ACL | 2021 | [![arXiv](https://img.shields.io/badge/arXiv-2012.15613-b31b1b.svg)](https://arxiv.org/abs/2012.15613) |
| **Language-Family Adapters for Multilingual Neural Machine Translation** | arXiv | 2022 | [![arXiv](https://img.shields.io/badge/arXiv-2209.15236-b31b1b.svg)](https://arxiv.org/abs/2209.15236) |
| **Vocabulary Expansion for LLM Adaptation** | arXiv | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2403.18071-b31b1b.svg)](https://arxiv.org/abs/2403.18071) |

---

## Byte-Level & Character-Level Models

An increasingly important research direction that removes or reduces the dependency on hand-crafted tokenization, operating directly on raw bytes or characters.

| Paper | Venue | Year | Links |
|-------|-------|------|-------|
| **ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models** | TACL | 2022 | [![arXiv](https://img.shields.io/badge/arXiv-2105.13626-b31b1b.svg)](https://arxiv.org/abs/2105.13626) [![GitHub](https://img.shields.io/badge/GitHub-Code-black)](https://github.com/google-research/byt5) |
| **CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation** | TACL | 2022 | [![arXiv](https://img.shields.io/badge/arXiv-2103.06874-b31b1b.svg)](https://arxiv.org/abs/2103.06874) |
| **Charformer: Fast Character Transformers via Gradient-based Subword Tokenization** | ICLR | 2022 | [![arXiv](https://img.shields.io/badge/arXiv-2106.12672-b31b1b.svg)](https://arxiv.org/abs/2106.12672) |
| **Character-Aware Neural Language Models** | AAAI | 2016 | [![arXiv](https://img.shields.io/badge/arXiv-1508.06615-b31b1b.svg)](https://arxiv.org/abs/1508.06615) |
| **MegaByte: Predicting Million-Byte Sequences with Multiscale Transformers** | NeurIPS | 2023 | [![arXiv](https://img.shields.io/badge/arXiv-2305.07185-b31b1b.svg)](https://arxiv.org/abs/2305.07185) |
| **Bytes Are All You Need: Transformers Operating Directly On File Bytes** | arXiv | 2023 | [![arXiv](https://img.shields.io/badge/arXiv-2306.00238-b31b1b.svg)](https://arxiv.org/abs/2306.00238) |
| **SpaceByte: Towards Deleting Tokenization from Large Language Modeling** | arXiv | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2404.14408-b31b1b.svg)](https://arxiv.org/abs/2404.14408) |
| **Byte Latent Transformer: Patches Scale Better Than Tokens** | Meta FAIR | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2412.09871-b31b1b.svg)](https://arxiv.org/abs/2412.09871) |
| **BYTE: Boosting Byte-level Transformers by Learning from N-gram** | arXiv | 2023 | [![arXiv](https://img.shields.io/badge/arXiv-2310.07710-b31b1b.svg)](https://arxiv.org/abs/2310.07710) |
| **Byte Pair Encoding for Byte-Level Machine Translation** | EMNLP | 2023 | [![arXiv](https://img.shields.io/badge/arXiv-2305.15974-b31b1b.svg)](https://arxiv.org/abs/2305.15974) |

---

## Tokenizer-Free Architectures

Models that learn their own segmentation end-to-end, eliminating the fixed tokenizer entirely.

| Paper | Venue | Year | Links |
|-------|-------|------|-------|
| **Pixel Language Models** | arXiv | 2023 | [![arXiv](https://img.shields.io/badge/arXiv-2207.06991-b31b1b.svg)](https://arxiv.org/abs/2207.06991) |
| **PIXEL: Language Modeling with Pixels** | ACL | 2023 | [![arXiv](https://img.shields.io/badge/arXiv-2207.06991-b31b1b.svg)](https://arxiv.org/abs/2207.06991) [![GitHub](https://img.shields.io/badge/GitHub-Code-black)](https://github.com/xplip/pixel) |
| **An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models** | ECCV | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2403.06764-b31b1b.svg)](https://arxiv.org/abs/2403.06764) |
| **Beyond Tokenization: A Multiscale Character-Level Language Model** | arXiv | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2404.03808-b31b1b.svg)](https://arxiv.org/abs/2404.03808) |
| **Learning to Tokenize for Generative Retrieval** | NeurIPS | 2023 | [![arXiv](https://img.shields.io/badge/arXiv-2304.04171-b31b1b.svg)](https://arxiv.org/abs/2304.04171) |

---

## Multilingual & Cross-Lingual Tokenization

Tokenization quality is highly uneven across languages — low-resource and morphologically rich languages are disproportionately affected by tokenizers trained primarily on English text.

| Paper | Venue | Year | Links |
|-------|-------|------|-------|
| **Do All Languages Cost the Same? Tokenization in the Era of Commercial Language Models** | EMNLP | 2023 | [![arXiv](https://img.shields.io/badge/arXiv-2305.13707-b31b1b.svg)](https://arxiv.org/abs/2305.13707) |
| **No Language Left Behind: Scaling Human-Centered Machine Translation** | arXiv | 2022 | [![arXiv](https://img.shields.io/badge/arXiv-2207.04672-b31b1b.svg)](https://arxiv.org/abs/2207.04672) |
| **Tokenizer Bias in Multilingual LLMs: Causes, Consequences, and Mitigations** | arXiv | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2406.16021-b31b1b.svg)](https://arxiv.org/abs/2406.16021) |
| **Language Model Tokenizers Introduce Unfairness Between Languages** | NeurIPS | 2023 | [![arXiv](https://img.shields.io/badge/arXiv-2305.15425-b31b1b.svg)](https://arxiv.org/abs/2305.15425) |
| **Overcoming the Curse of Multilinguality: Vocabulary Adaptation for Effective Multilingual LLMs** | arXiv | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2405.12818-b31b1b.svg)](https://arxiv.org/abs/2405.12818) |
| **The Tokenizer Playground: Understanding the Effect of Tokenization on Language Model Behavior** | EACL (Demo) | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2402.12571-b31b1b.svg)](https://arxiv.org/abs/2402.12571) |
| **Towards Building a Robust Toxicity Predictor** | ACL | 2023 | [![arXiv](https://img.shields.io/badge/arXiv-2404.08690-b31b1b.svg)](https://arxiv.org/abs/2404.08690) |
| **Adapting GPT, GPT-2 and BERT Language Models for Speech Recognition** | ASRU | 2019 | [![arXiv](https://img.shields.io/badge/arXiv-1906.02227-b31b1b.svg)](https://arxiv.org/abs/1906.02227) |
| **MorphPiece: A Linguistic Tokenizer for Large Language Models** | Findings of EMNLP | 2023 | [![arXiv](https://img.shields.io/badge/arXiv-2307.07262-b31b1b.svg)](https://arxiv.org/abs/2307.07262) |
| **How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models** | ACL | 2021 | [![arXiv](https://img.shields.io/badge/arXiv-2012.15613-b31b1b.svg)](https://arxiv.org/abs/2012.15613) |
| **Tokenization Impacts Multilingual Language Modeling: Assessing Vocabulary Allocation and Overlap Across Languages** | arXiv | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2406.01131-b31b1b.svg)](https://arxiv.org/abs/2406.01131) |

---

## Tokenization for Code & Structured Data

Code tokenization has unique challenges: preserving syntactic structure, handling indentation, supporting multiple programming languages, and dealing with identifiers.

| Paper | Venue | Year | Links |
|-------|-------|------|-------|
| **Codex: Evaluating Large Language Models Trained on Code** | arXiv | 2021 | [![arXiv](https://img.shields.io/badge/arXiv-2107.03374-b31b1b.svg)](https://arxiv.org/abs/2107.03374) |
| **InCoder: A Generative Model for Code Infilling and Synthesis** | ICLR | 2023 | [![arXiv](https://img.shields.io/badge/arXiv-2204.05999-b31b1b.svg)](https://arxiv.org/abs/2204.05999) |
| **StarCoder: May the Source Be With You!** | TMLR | 2023 | [![arXiv](https://img.shields.io/badge/arXiv-2305.06161-b31b1b.svg)](https://arxiv.org/abs/2305.06161) |
| **Code Llama: Open Foundation Models for Code** | arXiv | 2023 | [![arXiv](https://img.shields.io/badge/arXiv-2308.12950-b31b1b.svg)](https://arxiv.org/abs/2308.12950) |
| **Tokenization Matters: Navigating Data-Scarce Tokenization for Gender-Inclusive Language Technologies** | arXiv | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2407.13619-b31b1b.svg)](https://arxiv.org/abs/2407.13619) |
| **Learning to Compress Prompts with Gist Tokens** | NeurIPS | 2023 | [![arXiv](https://img.shields.io/badge/arXiv-2304.08467-b31b1b.svg)](https://arxiv.org/abs/2304.08467) |

---

## Tokenization & Arithmetic / Numerical Reasoning

One of the most surprising failure modes of LLMs — poor tokenization of numbers is a root cause of arithmetic errors.

| Paper | Venue | Year | Links |
|-------|-------|------|-------|
| **Impact of Tokenization on LLM Arithmetic** | arXiv | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2402.14903-b31b1b.svg)](https://arxiv.org/abs/2402.14903) |
| **Tokenization Counts: the Impact of Tokenization on Arithmetic in Frontier LLMs** | arXiv | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2402.14903-b31b1b.svg)](https://arxiv.org/abs/2402.14903) |
| **xVal: A Continuous Number Encoding for Large Language Models** | arXiv | 2023 | [![arXiv](https://img.shields.io/badge/arXiv-2310.02989-b31b1b.svg)](https://arxiv.org/abs/2310.02989) |
| **Teaching Arithmetic to Small Transformers** | ICLR | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2307.03381-b31b1b.svg)](https://arxiv.org/abs/2307.03381) |
| **Goat: Fine-tuned LLaMA Outperforms GPT-4 on Arithmetic Tasks** | arXiv | 2023 | [![arXiv](https://img.shields.io/badge/arXiv-2305.14201-b31b1b.svg)](https://arxiv.org/abs/2305.14201) |
| **Number Tokenization and Number Representation in Language Models** | arXiv | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2405.01714-b31b1b.svg)](https://arxiv.org/abs/2405.01714) |
| **Better Numerical Tokenization for Language Models** | arXiv | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2409.07898-b31b1b.svg)](https://arxiv.org/abs/2409.07898) |

---

## Multimodal Tokenization

Extending tokenization beyond text to images, audio, video, and other modalities.

| Paper | Venue | Year | Links |
|-------|-------|------|-------|
| **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale** *(ViT — patch tokenization)* | ICLR | 2021 | [![arXiv](https://img.shields.io/badge/arXiv-2010.11929-b31b1b.svg)](https://arxiv.org/abs/2010.11929) |
| **VQGAN: Taming Transformers for High-Resolution Image Synthesis** | CVPR | 2021 | [![arXiv](https://img.shields.io/badge/arXiv-2012.09841-b31b1b.svg)](https://arxiv.org/abs/2012.09841) |
| **Neural Discrete Representation Learning** *(VQ-VAE)* | NeurIPS | 2017 | [![arXiv](https://img.shields.io/badge/arXiv-1711.00937-b31b1b.svg)](https://arxiv.org/abs/1711.00937) |
| **Language Is Not All You Need: Aligning Perception with Language Models** *(Kosmos)* | NeurIPS | 2023 | [![arXiv](https://img.shields.io/badge/arXiv-2302.14045-b31b1b.svg)](https://arxiv.org/abs/2302.14045) |
| **Unified Tokenizer for Multimodal Generative AI** | arXiv | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2407.09985-b31b1b.svg)](https://arxiv.org/abs/2407.09985) |
| **Chameleon: Mixed-Modal Early-Fusion Foundation Models** | Meta | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2405.09818-b31b1b.svg)](https://arxiv.org/abs/2405.09818) |
| **EnCodec: High Fidelity Neural Audio Compression** *(audio tokenization)* | TMLR | 2023 | [![arXiv](https://img.shields.io/badge/arXiv-2210.13438-b31b1b.svg)](https://arxiv.org/abs/2210.13438) [![GitHub](https://img.shields.io/badge/GitHub-Code-black)](https://github.com/facebookresearch/encodec) |
| **SoundStream: An End-to-End Neural Audio Codec** | IEEE/ACM TASLP | 2022 | [![arXiv](https://img.shields.io/badge/arXiv-2107.03312-b31b1b.svg)](https://arxiv.org/abs/2107.03312) |
| **Cosmos Tokenizer: A Suite of Image and Video Tokenizers for Building Large Generative Models** | NVIDIA | 2025 | [![arXiv](https://img.shields.io/badge/arXiv-2501.03575-b31b1b.svg)](https://arxiv.org/abs/2501.03575) [![GitHub](https://img.shields.io/badge/GitHub-Code-black)](https://github.com/NVIDIA/Cosmos-Tokenizer) |

---

## Tokenizer Efficiency & Compression

Reducing the computational and memory cost of tokenization, or compressing token sequences for efficiency.

| Paper | Venue | Year | Links |
|-------|-------|------|-------|
| **Fewer Tokens for Better LLM Reasoning: Tokenization Efficiency and Inference** | arXiv | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2402.10930-b31b1b.svg)](https://arxiv.org/abs/2402.10930) |
| **In-Context Learning Creates Task Vectors** | Findings of EMNLP | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2310.15916-b31b1b.svg)](https://arxiv.org/abs/2310.15916) |
| **Learning to Compress Prompts with Gist Tokens** | NeurIPS | 2023 | [![arXiv](https://img.shields.io/badge/arXiv-2304.08467-b31b1b.svg)](https://arxiv.org/abs/2304.08467) |
| **LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models** | EMNLP | 2023 | [![arXiv](https://img.shields.io/badge/arXiv-2310.05736-b31b1b.svg)](https://arxiv.org/abs/2310.05736) [![GitHub](https://img.shields.io/badge/GitHub-Code-black)](https://github.com/microsoft/LLMLingua) |
| **Pathologies of Pre-trained Language Models in Few-shot Fine-tuning** | Findings of ACL | 2022 | [![arXiv](https://img.shields.io/badge/arXiv-2204.02340-b31b1b.svg)](https://arxiv.org/abs/2204.02340) |
| **Dynamic Token Merging for Efficient Vision Transformers** | NeurIPS | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2401.04228-b31b1b.svg)](https://arxiv.org/abs/2401.04228) |

---

## Tokenization Analysis & Evaluation

Understanding how tokenization choices affect model behavior, biases, and performance.

| Paper | Venue | Year | Links |
|-------|-------|------|-------|
| **On the Pitfalls of Measuring Emergent Communication** | AACL | 2019 | [![arXiv](https://img.shields.io/badge/arXiv-1903.05168-b31b1b.svg)](https://arxiv.org/abs/1903.05168) |
| **Tokenization Matters! Degrading Large Language Models through Challenging Their Tokenization** | arXiv | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2405.17067-b31b1b.svg)](https://arxiv.org/abs/2405.17067) |
| **What Do Tokens Know About Their Characters and How Do They Find Out?** | NAACL | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2206.02608-b31b1b.svg)](https://arxiv.org/abs/2206.02608) |
| **Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models** | arXiv | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2405.05417-b31b1b.svg)](https://arxiv.org/abs/2405.05417) [![GitHub](https://img.shields.io/badge/GitHub-Code-black)](https://github.com/Yonom/SolidGoldMagikarp) |
| **Counting Tokens: Evaluating Tokenizer Efficiency** | arXiv | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2402.03463-b31b1b.svg)](https://arxiv.org/abs/2402.03463) |
| **SolidGoldMagikarp: Anomalous Tokens in LLMs** | Lesswrong | 2023 | [Blog](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation) |
| **Language Model Tokenizers Introduce Unfairness Between Languages** | NeurIPS | 2023 | [![arXiv](https://img.shields.io/badge/arXiv-2305.15425-b31b1b.svg)](https://arxiv.org/abs/2305.15425) |

---

## Positional Encoding & Token Representation

While not tokenization per se, positional encodings determine how token positions are represented and are deeply intertwined with tokenizer design choices.

| Paper | Venue | Year | Links |
|-------|-------|------|-------|
| **Attention Is All You Need** *(Sinusoidal PE)* | NeurIPS | 2017 | [![arXiv](https://img.shields.io/badge/arXiv-1706.03762-b31b1b.svg)](https://arxiv.org/abs/1706.03762) |
| **RoFormer: Enhanced Transformer with Rotary Position Embedding** *(RoPE)* | Neurocomputing | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2104.09864-b31b1b.svg)](https://arxiv.org/abs/2104.09864) |
| **Train Short, Test Long: Attention with Linear Biases Enables Input Length Generalization** *(ALiBi)* | ICLR | 2022 | [![arXiv](https://img.shields.io/badge/arXiv-2108.12409-b31b1b.svg)](https://arxiv.org/abs/2108.12409) |
| **YaRN: Efficient Context Window Extension of Large Language Models** | ICLR | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2309.00071-b31b1b.svg)](https://arxiv.org/abs/2309.00071) |
| **Extending Context Window of Large Language Models via Positional Interpolation** | arXiv | 2023 | [![arXiv](https://img.shields.io/badge/arXiv-2306.15595-b31b1b.svg)](https://arxiv.org/abs/2306.15595) |
| **NTK-Aware Scaled RoPE** | Reddit/Blog | 2023 | [Post](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/) |

---

## Detokenization & Token Healing

The reverse process of tokenization, and techniques for fixing tokenization artifacts at generation boundaries.

| Paper | Venue | Year | Links |
|-------|-------|------|-------|
| **Token Healing in Language Models** | Guidance (Microsoft) | 2023 | [Blog](https://towardsdatascience.com/the-art-of-prompt-design-use-clear-syntax-4fc846c1acbd/) [![GitHub](https://img.shields.io/badge/GitHub-Code-black)](https://github.com/guidance-ai/guidance) |
| **Tokenization Repair in the Presence of Spelling Errors** | LREC-COLING | 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2404.03207-b31b1b.svg)](https://arxiv.org/abs/2404.03207) |

---

## Tools & Libraries

### Tokenizer Implementations

| Tool | Description | Links |
|------|-------------|-------|
| **🤗 Tokenizers** | Hugging Face's fast tokenizer library written in Rust. Supports BPE, WordPiece, and Unigram. The de facto standard. | [![GitHub](https://img.shields.io/badge/GitHub-Code-black)](https://github.com/huggingface/tokenizers) |
| **SentencePiece** | Google's language-independent tokenizer supporting BPE and Unigram. Used by T5, LLaMA, Gemma. | [![GitHub](https://img.shields.io/badge/GitHub-Code-black)](https://github.com/google/sentencepiece) |
| **tiktoken** | OpenAI's fast BPE tokenizer for GPT models. Written in Rust with Python bindings. | [![GitHub](https://img.shields.io/badge/GitHub-Code-black)](https://github.com/openai/tiktoken) |
| **minbpe** | Andrej Karpathy's minimal, educational BPE implementation. Perfect for learning how tokenizers work. | [![GitHub](https://img.shields.io/badge/GitHub-Code-black)](https://github.com/karpathy/minbpe) |
| **YouTokenToMe** | Unsupervised text tokenizer focused on speed. BPE with multithreaded training. | [![GitHub](https://img.shields.io/badge/GitHub-Code-black)](https://github.com/VKCOM/YouTokenToMe) |
| **tokenmonster** | Ungreedy tokenizer with a focus on efficiency. Vocabulary-optimized for compression. | [![GitHub](https://img.shields.io/badge/GitHub-Code-black)](https://github.com/alasdairforsythe/tokenmonster) |
| **MosaicML Streaming** | Efficient data loading for LLM training with built-in tokenization support. | [![GitHub](https://img.shields.io/badge/GitHub-Code-black)](https://github.com/mosaicml/streaming) |

### Visualization & Analysis

| Tool | Description | Links |
|------|-------------|-------|
| **Tiktokenizer** | Interactive web app for visualizing how different tokenizers segment text. | [![Web](https://img.shields.io/badge/Web-App-blue)](https://tiktokenizer.vercel.app/) [![GitHub](https://img.shields.io/badge/GitHub-Code-black)](https://github.com/dqbd/tiktokenizer) |
| **The Tokenizer Playground** | Compare tokenizers across different LLMs side-by-side. | [![Web](https://img.shields.io/badge/Web-App-blue)](https://huggingface.co/spaces/Xenova/the-tokenizer-playground) |
| **Token Counter** | OpenAI's official token counting tool. | [![Web](https://img.shields.io/badge/Web-App-blue)](https://platform.openai.com/tokenizer) |
| **Magikarp Token Explorer** | Explore under-trained and anomalous tokens in popular LLMs. | [![GitHub](https://img.shields.io/badge/GitHub-Code-black)](https://github.com/Yonom/SolidGoldMagikarp) |

---

## Tokenizers of Notable LLMs

A reference table of what tokenizer each major LLM uses and its vocabulary size.

| Model | Tokenizer Type | Vocab Size | Notes |
|-------|---------------|------------|-------|
| **GPT-2** | Byte-level BPE | 50,257 | First widely adopted byte-level BPE |
| **GPT-3** | Byte-level BPE | 50,257 | Same tokenizer as GPT-2 |
| **GPT-4 / GPT-4o** | Byte-level BPE (cl100k / o200k) | 100,256 / 199,997 | Significant vocab expansion; o200k adds multilingual tokens |
| **LLaMA / LLaMA 2** | SentencePiece (BPE) | 32,000 | Trained on multilingual data |
| **LLaMA 3 / 3.1** | tiktoken (BPE) | 128,256 | 4× vocab expansion from LLaMA 2 |
| **Mistral 7B** | SentencePiece (BPE) | 32,000 | Same tokenizer design as LLaMA |
| **Mixtral 8x7B** | SentencePiece (BPE) | 32,000 | Shared with Mistral |
| **Gemma / Gemma 2** | SentencePiece (BPE) | 256,000 | One of the largest vocabularies |
| **Gemini** | SentencePiece | 256,000 | Shared with Gemma family |
| **BERT** | WordPiece | 30,522 | Uses `##` prefix for continuation tokens |
| **T5** | SentencePiece (Unigram) | 32,000 | Unigram model, not BPE |
| **Qwen 2 / 2.5** | tiktoken (BPE) | 151,936 | Extended vocab for CJK languages |
| **DeepSeek V2/V3** | Byte-level BPE | 100,015 | Custom vocabulary |
| **Claude 3/4** | Byte-level BPE | ~100,000* | *Estimated; not publicly disclosed |
| **Phi-3** | tiktoken (BPE) | 32,064 | Compact vocabulary |
| **StarCoder 2** | Byte-level BPE | 49,152 | Code-optimized vocabulary |
| **Command R+** | SentencePiece (BPE) | 255,000 | Large multilingual vocab |
| **RWKV** | Custom (based on BPE) | 65,536 | Designed for RNN-like architecture |
| **Falcon** | Byte-level BPE | 65,024 | Focused on code and English |

---

## Blog Posts & Tutorials

| Title | Author/Source | Year | Link |
|-------|--------------|------|------|
| **Let's build the GPT Tokenizer** | Andrej Karpathy | 2024 | [![YouTube](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=zduSFxRajkE) |
| **How GPT Tokenizers Work** | Simon Willison | 2023 | [Blog](https://simonwillison.net/2023/Jun/8/gpt-tokenizers/) |
| **Summary of Tokenizers** | Hugging Face | 2023 | [Blog](https://huggingface.co/docs/transformers/tokenizer_summary) |
| **Byte Pair Encoding — The Dark Horse of Modern NLP** | Towards Data Science | 2021 | [Blog](https://towardsdatascience.com/byte-pair-encoding-the-dark-horse-of-modern-nlp-eb36c7df4f10/) |
| **SentencePiece Tokenizer Demystified** | Towards Data Science | 2021 | [Blog](https://towardsdatascience.com/sentencepiece-tokenizer-demystified-d0a3aac19b15/) |
| **A Deep Dive into the Wonderful World of LLM Tokenization** | Hamel Husain | 2024 | [Blog](https://hamel.dev/notes/llm/tokenization/index.html) |
| **Tokenizers: How Machines Read** | Hugging Face NLP Course | 2023 | [Course](https://huggingface.co/learn/nlp-course/chapter6/1) |
| **Understanding the BPE Tokenizer** | Lei Mao | 2023 | [Blog](https://leimao.github.io/blog/Byte-Pair-Encoding/) |
| **The SolidGoldMagikarp Phenomenon** | Jessica Rumbelow & Matthew Watkins | 2023 | [Blog](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation) |

---

## Talks & Videos

| Title | Speaker | Event | Link |
|-------|---------|-------|------|
| **Let's build the GPT Tokenizer** | Andrej Karpathy | YouTube | [![YouTube](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=zduSFxRajkE) |
| **BPE Tokenization - How LLMs Process Text** | 3Blue1Brown (Grant Sanderson) | YouTube | [![YouTube](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=YmSHMfpc5tQ) |
| **Tokenization: What Happens Before the LLM?** | Hugging Face | YouTube | [![YouTube](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=VPrHhzaGjRw) |

---

## Datasets & Benchmarks

| Resource | Description | Link |
|----------|-------------|------|
| **MTOB (Machine Translation from One Book)** | Evaluates tokenizer generalization to low-resource languages | [![arXiv](https://img.shields.io/badge/arXiv-2309.16575-b31b1b.svg)](https://arxiv.org/abs/2309.16575) |
| **Tokenizer Arena** | Community-driven tokenizer comparison platform | [![Web](https://img.shields.io/badge/Web-App-blue)](https://huggingface.co/spaces/Xenova/the-tokenizer-playground) |
| **UniMax** | Large-scale multilingual corpus for tokenizer training | [![Paper](https://img.shields.io/badge/Paper-Link-blue)](https://arxiv.org/abs/2304.06120) |

---

## Contributing

Contributions are welcome and encouraged! Here's how you can help:

1. **Add a paper**: Open a PR with the paper title, venue, year, and links (arXiv, GitHub if available)
2. **Add a tool**: Include the tool name, brief description, and repository link
3. **Fix errors**: If you spot incorrect information, please let us know
4. **Suggest categories**: If you think a new category would be useful, open an issue to discuss

Please ensure your contributions follow the existing format and include working links.

---

## Star History

If you find this repository useful, please consider giving it a ⭐ to help others discover it!

---

## License

[![CC0](https://licensebuttons.net/p/zero/1.0/88x31.png)](https://creativecommons.org/publicdomain/zero/1.0/)

This work is licensed under [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).
