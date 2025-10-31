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

These methods act on **when** KV data is executed, computed, or scheduled to improve latency and throughput. We divide these methods into three categories: KV-centric scheduling, pipelining & overlapping, and hardware-aware execution.

### KV-Centric Scheduling

|Year|Paper|Type|Venue|Code|
| -- | -- | -- | -- | -- |
| 2025 | TokenSelect: Efficient Long-Context Inference and Length Extrapolation for LLMs via Dynamic Token-Level KV Cache Selection [[Link](https://arxiv.org/pdf/2411.02886)] |  | EMNLP |[TokenSelect](https://github.com/pzs19/TokenSelect) [![stars](https://img.shields.io/github/stars/pzs19/TokenSelect?style=social)](https://github.com/pzs19/TokenSelect) |
| 2025 | RefreshKV: Updating Small KV Cache During Long-form Generation [[Link](https://aclanthology.org/2025.acl-long.1211.pdf)] |  | ACL |[RefreshKV](https://github.com/carriex/refreshkv) [![stars](https://img.shields.io/github/stars/carriex/refreshkv?style=social)](https://github.com/carriex/refreshkv) |
| 2025 | FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving [[Link](https://openreview.net/pdf?id=RXPofAsL8F)] |  | MLSys 🏆 **Outstanding Paper Award** |[FlashInfer](https://github.com/flashinfer-ai/flashinfer) 🌟 [![stars](https://img.shields.io/github/stars/flashinfer-ai/flashinfer?style=social)](https://github.com/flashinfer-ai/flashinfer) |
| 2025 | Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot [[Link](https://www.usenix.org/system/files/fast25-qin.pdf)] | Also belongs to HW-aware execution | FAST 🏆 **Best Paper Award** |[Mooncake](https://github.com/kvcache-ai/Mooncake) 🌟 [![stars](https://img.shields.io/github/stars/kvcache-ai/Mooncake?style=social)](https://github.com/kvcache-ai/Mooncake) |
| 2024 | Loki: Low-rank Keys for Efficient Sparse Attention [[Link](https://proceedings.neurips.cc/paper_files/paper/2024/file/1e027da6bec9ceb2ec37951ceeccae93-Paper-Conference.pdf)] |  | NeurIPS |[Loki](https://github.com/hpcgroup/loki) [![stars](https://img.shields.io/github/stars/hpcgroup/loki?style=social)](https://github.com/hpcgroup/loki) |
| 2024 | SGLang: Efficient Execution of Structured Language Model Programs [[Link](https://proceedings.neurips.cc/paper_files/paper/2024/file/724be4472168f31ba1c9ac630f15dec8-Paper-Conference.pdf)] |  | NeurIPS |[SGLang](https://github.com/sgl-project/sglang) 🌟 [![stars](https://img.shields.io/github/stars/sgl-project/sglang?style=social)](https://github.com/sgl-project/sglang) |
| 2024 | LoongServe: Efficiently Serving Long-Context Large Language Models with Elastic Sequence Parallelism [[Link](https://arxiv.org/pdf/2404.09526)] |  | SOSP |[LoongServe](https://github.com/LoongServe/LoongServe) [![stars](https://img.shields.io/github/stars/LoongServe/LoongServe?style=social)](https://github.com/LoongServe/LoongServe) |
| 2024 | Fast Inference for Augmented Large Language Models [[Link](https://arxiv.org/pdf/2404.09526)] |  | arXiv | |
| 2024 | LayerKV: Optimizing Large Language Model Serving with Layer-wise KV Cache Management [[Link](https://arxiv.org/pdf/2410.00428)] | Also belongs to memory hierarchy KV orchestration (spatial) | arXiv | |
| 2024 | SparQ Attention: Bandwidth-Efficient LLM Inference [[Link](https://openreview.net/pdf?id=OS5dqxmmtl)] |  | ICML |[SparQ Attention](https://github.com/graphcore-research/llm-inference-research/tree/2024-05-sparq) [![stars](https://img.shields.io/github/stars/graphcore-research/llm-inference-research?style=social)](https://github.com/graphcore-research/llm-inference-research/tree/2024-05-sparq) |
| 2024 | QUEST: Query-Aware Sparsity for Efficient Long-Context LLM Inference [[Link](https://openreview.net/pdf?id=KzACYw0MTV)] |  | ICML |[Quest](https://github.com/mit-han-lab/quest) [![stars](https://img.shields.io/github/stars/mit-han-lab/quest?style=social)](https://github.com/mit-han-lab/quest) |
| 2024 | MuxServe: Flexible Spatial-Temporal Multiplexing for Multiple LLM Serving [[Link](https://openreview.net/pdf?id=R0SoZvqXyQ)] | Also belongs to HW-aware execution | ICML |[MuxServe](https://github.com/hao-ai-lab/MuxServe) [![stars](https://img.shields.io/github/stars/hao-ai-lab/MuxServe?style=social)](https://github.com/hao-ai-lab/MuxServe) |
| 2024 | Preble: Efficient Distributed Prompt Scheduling for LLM Serving [[Link](https://openreview.net/pdf?id=meKEKDhdnx)] |  | ICLR |[Preble](https://github.com/WukLab/preble) [![stars](https://img.shields.io/github/stars/WukLab/preble?style=social)](https://github.com/WukLab/preble) |
| 2024 | Inference without interference: Disaggregate LLM inference for mixed downstream workloads [[Link](https://arxiv.org/pdf/2401.11181)] | Also belongs to HW-aware execution | arXiv | |

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>

### Pipelining & Overlapping

|Year|Paper|Type|Venue|Code|
| -- | -- | -- | -- | -- |
| 2025 | KVPR: Efficient LLM inference with i/o-aware KV cache partial recomputation [[Link](https://aclanthology.org/2025.findings-acl.997.pdf)] |  | ACL Findings |[KVPR](https://github.com/chaoyij/KVPR) [![stars](https://img.shields.io/github/stars/chaoyij/KVPR?style=social)](https://github.com/chaoyij/KVPR) |
| 2025 | PRESERVE: Prefetching Model Weights and KV-Cache in Distributed LLM Serving [[Link](https://arxiv.org/pdf/2501.08192)] | Also belongs to memory hierarchy KV orchestration (spatial) | arXiv ||
| 2025 | NEO: Saving GPU Memory Crisis with CPU Offloading for Online LLM Inference [[Link](https://openreview.net/pdf?id=umgy9tWBLA)] | Also belongs to HW-aware execution | MLSys |[NEO](https://github.com/NEO-MLSys25/NEO) [![stars](https://img.shields.io/github/stars/NEO-MLSys25/NEO?style=social)](https://github.com/NEO-MLSys25/NEO) |
| 2025 | Accelerating LLM Inference Throughput via Asynchronous KV Cache Prefetching [[Link](https://arxiv.org/pdf/2504.06319)] | Also belongs to memory hierarchy KV orchestration (spatial) | arXiv ||
| 2024 | Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention [[Link](https://www.usenix.org/system/files/atc24-gao-bin-cost.pdf)] | Also belongs to memory hierarchy KV orchestration (spatial) | ATC ||
| 2024 | FastDecode: High-Throughput GPU-Efficient LLM Serving using Heterogeneous Pipelines [[Link](https://arxiv.org/pdf/2403.11421)] | Also belongs to HW-aware execution | arXiv ||
| 2024 | Improving Throughput-Oriented LLM Inference with CPU Computations [[Link](https://dl.acm.org/doi/pdf/10.1145/3656019.3676949)] | Also belongs to HW-aware execution | PACT |[Heterogen](https://gitlab.csap.snu.ac.kr/research/heterogen)|

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>

### Hardware-aware Execution

#### Disaggregated Inference

|Year|Paper|Type|Venue|Code|
| -- | -- | -- | -- | -- |
| 2025 | Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot [[Link](https://www.usenix.org/system/files/fast25-qin.pdf)] | Also belongs to KV-centric scheduling | FAST 🏆 **Best Paper Award** |[Mooncake](https://github.com/kvcache-ai/Mooncake) [![stars](https://img.shields.io/github/stars/kvcache-ai/Mooncake?style=social)](https://github.com/kvcache-ai/Mooncake) |
| 2024 | DéjàVu: KV-cache Streaming for Fast, Fault-tolerant Generative LLM Serving [[Link](https://openreview.net/pdf?id=AbGbGZFYOD)] | Also belongs to memory hierarchy KV orchestration (spatial) | ICML |[DéjàVu](https://github.com/msr-fiddle/dejavu) [![stars](https://img.shields.io/github/stars/msr-fiddle/dejavu?style=social)](https://github.com/msr-fiddle/dejavu) |
| 2024 | MuxServe: Flexible Spatial-Temporal Multiplexing for Multiple LLM Serving [[Link](https://openreview.net/pdf?id=R0SoZvqXyQ)] | Also belongs to KV-centric scheduling | ICML |[MuxServe](https://github.com/hao-ai-lab/MuxServe) [![stars](https://img.shields.io/github/stars/hao-ai-lab/MuxServe?style=social)](https://github.com/hao-ai-lab/MuxServe) |
| 2024 | DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving [[Link](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf)] | Also belongs to compute device KV orchestration (spatial) | OSDI |[DistServe](https://github.com/LLMServe/DistServe) [![stars](https://img.shields.io/github/stars/LLMServe/DistServe?style=social)](https://github.com/LLMServe/DistServe) |
| 2024 | Infinite-LLM: Efficient LLM Service for Long Context with DistAttention and Distributed KVCache [[Link](https://arxiv.org/pdf/2401.02669)] | Also belongs to compute device KV orchestration (spatial) | aXiv ||
| 2024 | Splitwise: Efficient generative LLM inference using phase splitting [[Link](https://arxiv.org/abs/2311.18677)] | Also belongs to compute device KV orchestration (spatial) | ISCA |[Splitwise](https://github.com/vllm-project/vllm/pull/2809) (integrated into vLLM) |
| 2024 | Inference without interference: Disaggregate LLM inference for mixed downstream workloads [[Link](https://arxiv.org/pdf/2401.11181)] | Also belongs to KV-centric scheduling | arXiv | |


#### Compute Offloading

|Year|Paper|Type|Venue|Code|
| -- | -- | -- | -- | -- |
| 2025 | NEO: Saving GPU Memory Crisis with CPU Offloading for Online LLM Inference [[Link](https://openreview.net/pdf?id=umgy9tWBLA)] | Also belongs to pipelining & overlapping | MLSys |[NEO](https://github.com/NEO-MLSys25/NEO) [![stars](https://img.shields.io/github/stars/NEO-MLSys25/NEO?style=social)](https://github.com/NEO-MLSys25/NEO) |
| 2025 | MagicPIG: LSH Sampling for Efficient LLM Generation [[Link](https://openreview.net/pdf?id=ALzTQUgW8a)] |  | ICLR 💡 **Spotlight** |[MagicPIG](https://github.com/Infini-AI-Lab/MagicPIG) [![stars](https://img.shields.io/github/stars/Infini-AI-Lab/MagicPIG?style=social)](https://github.com/Infini-AI-Lab/MagicPIG) |
| 2025 | PAPI: Exploiting Dynamic Parallelism in Large Language Model Decoding with a Processing-In-Memory-Enabled Computing System [[Link](https://dl.acm.org/doi/pdf/10.1145/3676641.3716009)] |  | ASPLOS | |
| 2024 | TwinPilots: A New Computing Paradigm for GPU-CPU Parallel LLM Inference [[Link](https://dl.acm.org/doi/pdf/10.1145/3688351.3689164)] |  | SYSTOR | |
| 2024 | InstInfer: In-Storage Attention Offloading for Cost-Effective Long-Context LLM Inference [[Link](https://arxiv.org/pdf/2409.04992)] | Also belongs to compute device KV orchestration (spatial) | arXiv | |
| 2024 | AttAcc! unleashing the power of PIM for batched transformer-based generative model inference [[Link](https://dl.acm.org/doi/10.1145/3620665.3640422)] | Also belongs to compute device KV orchestration (spatial) | ASPLOS |[AttAcc](https://github.com/scale-snu/attacc_simulator) [![stars](https://img.shields.io/github/stars/scale-snu/attacc_simulator?style=social)](https://github.com/scale-snu/attacc_simulator) |
| 2024 | FastDecode: High-Throughput GPU-Efficient LLM Serving using Heterogeneous Pipelines [[Link](https://arxiv.org/pdf/2403.11421)] | Also belongs to pipelining & overlapping | arXiv ||
| 2024 | Improving Throughput-Oriented LLM Inference with CPU Computations [[Link](https://dl.acm.org/doi/pdf/10.1145/3656019.3676949)] | Also belongs to pipelining & overlapping | PACT |[Heterogen](https://gitlab.csap.snu.ac.kr/research/heterogen)|


<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>

---

## Spatial — Placement & Migration

These works optimize **where** KV data is stored or transferred to balance memory and I/O pressure. We divide these methods into two categories: memory hierarchy KV orchestration, and compute device KV orchestration.


### Memory Hierarchy KV Orchestration

#### Cross-device Memory Hierarchy

|Year|Paper|Type|Venue|Code|
| -- | -- | -- | -- | -- |
| 2025 | RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval [[Link](https://openreview.net/pdf?id=8z3cOVER4z)] |  | NeurIPS |[RetrievalAttention](https://github.com/microsoft/RetrievalAttention) [![stars](https://img.shields.io/github/stars/microsoft/RetrievalAttention?style=social)](https://github.com/microsoft/RetrievalAttention) |
| 2025 | RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation [[Link](https://dl.acm.org/doi/pdf/10.1145/3768628)] |  | TOCS | |
| 2025 | ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference [[Link](https://openreview.net/pdf?id=oa7MYAO6h6)] | Also belongs to KV cache compression (structural) | ICML 💡 **Spotlight** |[ShadowKV](https://github.com/ByteDance-Seed/ShadowKV) [![stars](https://img.shields.io/github/stars/ByteDance-Seed/ShadowKV?style=social)](https://github.com/ByteDance-Seed/ShadowKV) |
| 2025 | PQCache: Product Quantization-based KVCache for Long Context LLM Inference [[Link](https://arxiv.org/pdf/2407.12820)] |  | SIGMOD |[PQCache](https://github.com/HugoZHL/PQCache) [![stars](https://img.shields.io/github/stars/HugoZHL/PQCache?style=social)](https://github.com/HugoZHL/PQCache) |
| 2025 | ClusterKV: Manipulating LLM KV Cache in Semantic Space for Recallable Compression [[Link](https://arxiv.org/abs/2412.03213)] |  | DAC |[ClusterKV](https://github.com/sjtu-zhao-lab/ClusterKV) [![stars](https://img.shields.io/github/stars/sjtu-zhao-lab/ClusterKV?style=social)](https://github.com/sjtu-zhao-lab/ClusterKV) |
| 2025 | Stateful Large Language Model Serving with Pensieve [[Link](https://arxiv.org/pdf/2312.05516)] |  | EuroSys | |
| 2024 | InfLLM: Training-Free Long-Context Extrapolation for LLMs with an Efficient Context Memory [[Link](https://proceedings.neurips.cc/paper_files/paper/2024/file/d842425e4bf79ba039352da0f658a906-Paper-Conference.pdf)] |  | NeurIPS |[InfLLM](https://github.com/thunlp/InfLLM) [![stars](https://img.shields.io/github/stars/thunlp/InfLLM?style=social)](https://github.com/thunlp/InfLLM) |
| 2024 | FastSwitch: Optimizing Context Switching Efficiency in Fairness-aware Large Language Model Serving [[Link](https://arxiv.org/pdf/2411.18424)] |  | arXiv || 
| 2024 | LayerKV: Optimizing Large Language Model Serving with Layer-wise KV Cache Management [[Link](https://arxiv.org/pdf/2410.00428)] | Also belongs to KV-centric scheduling (temporal) | arXiv | |
| 2024 | DéjàVu: KV-cache Streaming for Fast, Fault-tolerant Generative LLM Serving [[Link](https://openreview.net/pdf?id=AbGbGZFYOD)] | Also belongs to HW-aware execution (temporal) | ICML |[DéjàVu](https://github.com/msr-fiddle/dejavu) [![stars](https://img.shields.io/github/stars/msr-fiddle/dejavu?style=social)](https://github.com/msr-fiddle/dejavu) |
| 2024 | Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention [[Link](https://www.usenix.org/system/files/atc24-gao-bin-cost.pdf)] | Also belongs to pipelining & overlapping (temporal) | ATC ||
| 2024 | InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management [[Link](https://www.usenix.org/system/files/osdi24-lee.pdf)] |  | OSDI |[InfiniGen](https://github.com/snu-comparch/InfiniGen) [![stars](https://img.shields.io/github/stars/snu-comparch/InfiniGen?style=social)](https://github.com/snu-comparch/InfiniGen) |
| 2024 | ALISA: Accelerating Large Language Model Inference via Sparsity-Aware KV Caching [[Link](https://arxiv.org/pdf/2403.17312)] |  | ISCA | |
| 2023 | Distributed Inference and Fine-tuning of Large Language Models Over The Internet [[Link](https://openreview.net/pdf?id=XmN7ZNbUAe)] | Also belongs to compute device KV orchestration | NeurIPS |[FastServe](https://github.com/LLMServe/FastServe) [![stars](https://img.shields.io/github/stars/LLMServe/FastServe?style=social)](https://github.com/LLMServe/FastServe) |
| 2023 | FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU [[Link](https://openreview.net/pdf?id=RRntzKrBTp)] | Also belongs to KV cache compression (structural) | ICML |[FlexLLMGen](https://github.com/FMInference/FlexLLMGen) 🌟 [![stars](https://img.shields.io/github/stars/FMInference/FlexLLMGen?style=social)](https://github.com/FMInference/FlexLLMGen) |


#### Intra-GPU Memory Hierarchy

|Year|Paper|Type|Venue|Code|
| -- | -- | -- | -- | -- |
| 2025 | PRESERVE: Prefetching Model Weights and KV-Cache in Distributed LLM Serving [[Link](https://arxiv.org/pdf/2501.08192)] | Also belongs to pipelining & overlapping (temporal) | arXiv ||
| 2025 | Accelerating LLM Inference Throughput via Asynchronous KV Cache Prefetching [[Link](https://arxiv.org/pdf/2504.06319)] | Also belongs to pipelining & overlapping (temporal) | arXiv ||


<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>



### Compute Device KV Orchestration

|Year|Paper|Type|Venue|Code|
| -- | -- | -- | -- | -- |
| 2024 | InstInfer: In-Storage Attention Offloading for Cost-Effective Long-Context LLM Inference [[Link](https://arxiv.org/pdf/2409.04992)] | Also belongs to HW-aware execution (temporal) | arXiv | |
| 2024 | CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving [[Link](https://dl.acm.org/doi/pdf/10.1145/3651890.3672274)] | Also belongs to KV cache compression (structural) | SIGCOMM |[CacheGen](https://github.com/UChi-JCL/CacheGen) [![stars](https://img.shields.io/github/stars/UChi-JCL/CacheGen?style=social)](https://github.com/UChi-JCL/CacheGen) |
| 2024 | DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving [[Link](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf)] | Also belongs to HW-aware execution (temporal) | OSDI |[DistServe](https://github.com/LLMServe/DistServe) [![stars](https://img.shields.io/github/stars/LLMServe/DistServe?style=social)](https://github.com/LLMServe/DistServe) |
| 2024 | Infinite-LLM: Efficient LLM Service for Long Context with DistAttention and Distributed KVCache [[Link](https://arxiv.org/pdf/2401.02669)] | Also belongs to HW-aware execution (temporal) | aXiv ||
| 2024 | Splitwise: Efficient generative LLM inference using phase splitting [[Link](https://arxiv.org/abs/2311.18677)] | Also belongs to HW-aware execution (temporal) | ISCA |[Splitwise](https://github.com/vllm-project/vllm/pull/2809) (integrated into vLLM) |
| 2024 | AttAcc! unleashing the power of PIM for batched transformer-based generative model inference [[Link](https://dl.acm.org/doi/10.1145/3620665.3640422)] | Also belongs to HW-aware execution (temporal) | ASPLOS |[AttAcc](https://github.com/scale-snu/attacc_simulator) [![stars](https://img.shields.io/github/stars/scale-snu/attacc_simulator?style=social)](https://github.com/scale-snu/attacc_simulator) |
| 2023 | Distributed Inference and Fine-tuning of Large Language Models Over The Internet [[Link](https://openreview.net/pdf?id=XmN7ZNbUAe)] | Also belongs to memory hierarchy KV orchestration | NeurIPS |[FastServe](https://github.com/LLMServe/FastServe) [![stars](https://img.shields.io/github/stars/LLMServe/FastServe?style=social)](https://github.com/LLMServe/FastServe) |


<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>

---

## Structural — Representation & Retention

These methods target **how** KV data is represented and maintained for memory efficiency. We divide these methods into two categories: KV cache compression, and KV cache retention management.

### KV Cache Compression
#### Quantization


Still work in progress.



#### Low-rank Approximation

Still work in progress.



#### Structural Compression

Still work in progress.



<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>

### KV Cache Retention Management
#### Allocation & Reuse

Still work in progress.


#### Eviction

Still work in progress.



<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>

---

## Cross-behavior Synergies

Still work in progress.

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>

---


## Behavior-objective Synergies

Still work in progress.


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