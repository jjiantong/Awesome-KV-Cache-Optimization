<h1 align="center">Awesome KV Cache Optimization</h1>

<!-- <p align="center">
    <b> Curated collection of papers on system-aware, serving-time, KV-centric techniques.</b>
</p> -->

<p align="center">
    <img src="assets/awesome-cover.png" width="80%"  style="align:center;"/>
</p>

This repository aims to record papers of system-aware, serving-time, KV-centric optimization methods that improve system metrics without retraining or architecture modification (which we call this scope ***sKis***). This serves as supplementary materials for our survey paper:

> **Towards Efficient Large Language Model Serving: A Survey on System-Aware KV Cache Optimization**  
> 📄🔗 [TechRxiv Preprint](https://doi.org/10.36227/techrxiv.176046306.66521015/v1) (DOI: 10.36227/techrxiv.176046306.66521015/v1)  
> 🧑‍💻👩‍💻 *[Jiantong Jiang](https://jjiantong.github.io/)<sup>1</sup>, [Peiyu Yang](https://ypeiyu.github.io/)<sup>1</sup>, [Rui Zhang](https://www.ruizhang.info/)<sup>2</sup>, [Feng Liu](https://fengliu90.github.io/)<sup>1</sup>*  
> <sup>1</sup>The University of Melbourne, <sup>2</sup>Huazhong University of Science and Technology

The survey and the repository are **still work in progress**.

---

<a name="readme-top"></a>

## Overview

The real bottleneck in LLM inference serving is often the **KV cache**, especially under long contexts and high concurrency. Our survey systematizes recent advances through a distinct **system behavior-oriented taxonomy**, which organizes existing efforts into three behavioral dimensions:\
🔷 **Temporal** — when is KV cache accessed or computed?\
🔷 **Spatial** — where is KV cache placed and migrated?\
🔷 **Structural** — how is KV cache represented and managed?

🧠 Grounded in this taxonomy, we analyze **cross-behavior synergies** and **behavior–objective effects**, revealing overlooked regions and concrete open challenges. 


## Quick Index

- [Temporal — Execution \& Scheduling](#temporal--execution--scheduling)
  - [KV-Centric Scheduling](#kv-centric-scheduling)
  - [Pipelining \& Overlapping](#pipelining--overlapping)
  - [Hardware-Aware Execution](#hardware-aware-execution) 
- [Spatial — Placement \& Migration](#spatial--placement--migration)
  - [Memory Hierarchy KV Orchestration](#memory-hierarchy-kv-orchestration)
  - [Compute Device KV Orchestration](#compute-device-kv-orchestration)
- [Structural — Representation \& Retention](#structural--representation--retention)
  - [KV Cache Compression](#kv-cache-compression) (including quantization, low-rank approximation, and structural compression)
  - [KV Cache Retention Management](#kv-cache-retention-management) (including allocation, reuse, and eviction)
- [Cross-behavior Synergies](#cross-behavior-synergies)
- [Behavior-objective Effects](#behavior-objective-effects)
- [Citation](#citation)
- [Contributing](#contributing)

---

## Temporal — Execution & Scheduling

These methods act on **when** KV data is executed, computed, or scheduled to improve latency and throughput.

### KV-Centric Scheduling

|Year|Title|Type|Venue|Paper|Code|
| -- | -- | -- | -- | -- | -- |
| 2025 | TokenSelect: Efficient Long-Context Inference and Length Extrapolation for LLMs via Dynamic Token-Level KV Cache Selection |  | EMNLP | [Link](https://arxiv.org/pdf/2411.02886) |[TokenSelect](https://github.com/pzs19/TokenSelect) [![stars](https://img.shields.io/github/stars/pzs19/TokenSelect?style=social)](https://github.com/pzs19/TokenSelect) |
| 2025 | RefreshKV: Updating Small KV Cache During Long-form Generation |  | ACL | [Link](https://aclanthology.org/2025.acl-long.1211.pdf) |[RefreshKV](https://github.com/carriex/refreshkv) [![stars](https://img.shields.io/github/stars/carriex/refreshkv?style=social)](https://github.com/carriex/refreshkv) |
| 2025 | FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving |  | MLSys 🏆 **Outstanding Paper Award** | [Link](https://openreview.net/pdf?id=RXPofAsL8F) |[FlashInfer](https://github.com/flashinfer-ai/flashinfer) [![stars](https://img.shields.io/github/stars/flashinfer-ai/flashinfer?style=social)](https://github.com/flashinfer-ai/flashinfer) |
| 2025 | Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot |  | FAST 🏆 **Best Paper Award** | [Link](https://www.usenix.org/system/files/fast25-qin.pdf) |[Mooncake](https://github.com/kvcache-ai/Mooncake) [![stars](https://img.shields.io/github/stars/kvcache-ai/Mooncake?style=social)](https://github.com/kvcache-ai/Mooncake) |
| 2024 | Loki: Low-rank Keys for Efficient Sparse Attention |  | NeurIPS | [Link](https://proceedings.neurips.cc/paper_files/paper/2024/file/1e027da6bec9ceb2ec37951ceeccae93-Paper-Conference.pdf) |[Loki](https://github.com/hpcgroup/loki) [![stars](https://img.shields.io/github/stars/hpcgroup/loki?style=social)](https://github.com/hpcgroup/loki) |
| 2024 | SGLang: Efficient Execution of Structured Language Model Programs |  | NeurIPS | [Link](https://proceedings.neurips.cc/paper_files/paper/2024/file/724be4472168f31ba1c9ac630f15dec8-Paper-Conference.pdf) |[SGLang](https://github.com/sgl-project/sglang) [![stars](https://img.shields.io/github/stars/sgl-project/sglang?style=social)](https://github.com/sgl-project/sglang) |
| 2024 | LoongServe: Efficiently Serving Long-Context Large Language Models with Elastic Sequence Parallelism |  | SOSP | [Link](https://arxiv.org/pdf/2404.09526) |[LoongServe](https://github.com/LoongServe/LoongServe) [![stars](https://img.shields.io/github/stars/LoongServe/LoongServe?style=social)](https://github.com/LoongServe/LoongServe) |
| 2024 | Fast Inference for Augmented Large Language Models |  | arXiv | [Link](https://arxiv.org/pdf/2404.09526) | |
| 2024 | LayerKV: Optimizing Large Language Model Serving with Layer-wise KV Cache Management |  | arXiv | [Link](https://arxiv.org/pdf/2410.00428) | |
| 2024 | SparQ Attention: Bandwidth-Efficient LLM Inference |  | ICML | [Link](https://openreview.net/pdf?id=OS5dqxmmtl) |[SparQ Attention](https://github.com/graphcore-research/llm-inference-research/tree/2024-05-sparq) [![stars](https://img.shields.io/github/stars/graphcore-research/llm-inference-research?style=social)](https://github.com/graphcore-research/llm-inference-research/tree/2024-05-sparq) |
| 2024 | QUEST: Query-Aware Sparsity for Efficient Long-Context LLM Inference |  | ICML | [Link](https://openreview.net/pdf?id=KzACYw0MTV) |[Quest](https://github.com/mit-han-lab/quest) [![stars](https://img.shields.io/github/stars/mit-han-lab/quest?style=social)](https://github.com/mit-han-lab/quest) |
| 2024 | MuxServe: Flexible Spatial-Temporal Multiplexing for Multiple LLM Serving |  | ICML | [Link](https://openreview.net/pdf?id=R0SoZvqXyQ) |[MuxServe](https://github.com/hao-ai-lab/MuxServe) [![stars](https://img.shields.io/github/stars/hao-ai-lab/MuxServe?style=social)](https://github.com/hao-ai-lab/MuxServe) |
| 2024 | Preble: Efficient Distributed Prompt Scheduling for LLM Serving |  | ICLR | [Link](https://openreview.net/pdf?id=meKEKDhdnx) |[Preble](https://github.com/WukLab/preble) [![stars](https://img.shields.io/github/stars/WukLab/preble?style=social)](https://github.com/WukLab/preble) |
| 2024 | Inference without interference: Disaggregate LLM inference for mixed downstream workloads |  | arXiv | [Link](https://arxiv.org/pdf/2401.11181) | |


<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>

### Pipelining & Overlapping

|Year|Title|Type|Venue|Paper|Code|
| -- | -- | -- | -- | -- | -- |
| 2025 | KVPR: Efficient LLM inference with i/o-aware KV cache partial recomputation |  | ACL Findings | [Link](https://aclanthology.org/2025.findings-acl.997.pdf) |[KVPR](https://github.com/chaoyij/KVPR) [![stars](https://img.shields.io/github/stars/chaoyij/KVPR?style=social)](https://github.com/chaoyij/KVPR) |


<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>

### Hardware-aware Execution

#### Disaggregated Inference

#### Compute Offloading

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>

---

## Spatial — Placement & Migration

These works optimize **where** KV data is stored or transferred to balance memory and I/O pressure.

### Memory Hierarchy KV Orchestration

#### Cross-device Memory Hierarchy

#### Intra-GPU Memory Hierarchy

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>

### Compute Device KV Orchestration

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>

---

## Structural — Representation & Retention

These methods target **how** KV data is represented and maintained for memory efficiency.

### KV Cache Compression
#### Quantization

|Year|Title|Type|Venue|Paper|Code|
| -- | -- | -- | -- | -- | -- |
| 2025 | Accurate KV Cache Quantization with Outlier Tokens Tracing |  | ACL | [Link](https://aclanthology.org/2025.acl-long.631.pdf) |[OTT](https://github.com/yisunlp/OTT) [![stars](https://img.shields.io/github/stars/yisunlp/OTT?style=social)](https://github.com/yisunlp/OTT) |

#### Low-rank Approximation

#### Structural Compression

|Year|Title|Type|Venue|Paper|Code|
| -- | -- | -- | -- | -- | -- |
| 2025 | ClusterAttn: KV Cache Compression under Intrinsic Attention Clustering |  | ACL | [Link](https://aclanthology.org/2025.acl-long.703.pdf) | |


<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>

### KV Cache Retention Management
#### Allocation & Reuse

#### Eviction

|Year|Title|Type|Venue|Paper|Code|
| -- | -- | -- | -- | -- | -- |
| 2025 | LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models |  | ICML | [Link](https://openreview.net/pdf?id=SDjZtxDo35) |[LaCache](https://github.com/GATECH-EIC/LaCache) [![stars](https://img.shields.io/github/stars/GATECH-EIC/LaCache?style=social)](https://github.com/GATECH-EIC/LaCache) |

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>

---

## Cross-behavior Synergies


<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>

---


## Behavior-objective Synergies

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>





## Citation

The survey will be updated regularly. If you find this survey helpful for your work, please consider citing it.
```
@article{jiang2025towards,
  title = {Towards Efficient Large Language Model Serving: A Survey on System-Aware KV Cache Optimization},
  author = {Jiang, Jiantong and Yang, Peiyu and Zhang, Rui and Liu, Feng},
  journal = {Authorea Preprints},
  year = {2025},
  publisher = {Authorea},
  url = {http://dx.doi.org/10.36227/techrxiv.176046306.66521015/v1},
  doi = {10.36227/techrxiv.176046306.66521015/v1},
}
```


## Contributing

If you would like to include other papers in this survey and repository, please feel free to contact us via email or open an issue with the paper's title, category, and a brief summary highlighting its key techniques and contributions. Other comments regarding this repository or survey are also highly welcome. Thank you!


<!-- ### Contributors

<a href="https://github.com/atfortes/Awesome-KV-Cache-Optimization/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=atfortes/Awesome-KV-Cache-Optimization" />
</a> -->