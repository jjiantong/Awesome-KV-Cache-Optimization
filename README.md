<h1 align="center">Awesome KV Cache Optimization</h1>

<div align="center">

![](https://img.shields.io/github/stars/jjiantong/Awesome-KV-Cache-Optimization)
![](https://img.shields.io/github/watchers/jjiantong/Awesome-KV-Cache-Optimization)
![](https://img.shields.io/github/last-commit/jjiantong/Awesome-KV-Cache-Optimization?color=green)
![](https://img.shields.io/badge/PRs-Welcome-blue)
[![DOI](https://img.shields.io/badge/DOI-10.36227%2Ftechrxiv.176046306.66521015%2Fv1-yellow?logo=doi)](https://doi.org/10.36227/techrxiv.176046306.66521015/v1)

</div>

<div align="center">

**[<a href="https://doi.org/10.36227/techrxiv.176046306.66521015/v1">TechRxiv</a>]** **[<a href="https://www.linkedin.com/feed/update/urn:li:activity:7384388868407529472/">LinkedIn</a>]**

</div>


This repository is for our survey paper:

> **[Towards Efficient Large Language Model Serving: A Survey on System-Aware KV Cache Optimization](https://doi.org/10.36227/techrxiv.176046306.66521015/v1)**  
> *[Jiantong Jiang](https://jjiantong.github.io/)<sup>1</sup>, [Peiyu Yang](https://ypeiyu.github.io/)<sup>1</sup>, [Rui Zhang](https://www.ruizhang.info/)<sup>2</sup>, [Feng Liu](https://fengliu90.github.io/)<sup>1</sup>*  
> <sup>1</sup>The University of Melbourne, <sup>2</sup>Huazhong University of Science and Technology

<p align="center">
    <img src="assets/awesome-cover.png" width="90%"  style="align:center;"/>
</p>


This repository aims to record papers of system-aware, serving-time, KV-centric optimization methods that improve system metrics without retraining or architecture modification (which we call this scope ***sKis***). We systematize recent advances through a distinct **system behavior-oriented taxonomy**, organizing existing efforts into three behavioral dimensions:\
🔷 **Temporal** — when is KV cache accessed or computed?\
🔷 **Spatial** — where is KV cache placed and migrated?\
🔷 **Structural** — how is KV cache represented and managed?

🧠 Grounded in this taxonomy, we analyze **cross-behavior synergies** and **behavior–objective effects**, revealing overlooked regions and concrete open challenges. 


### Contributing

The survey and the repository are **still work in progress** and will be updated regularly. 

🙋 If you would like to include your paper in this survey and repository, please feel free to submit a pull request or open an issue with the paper's title and a brief summary highlighting its key techniques. You can also contact us via email. Please let us know if you find out a mistake or have any suggestions! We greatly appreciate your feedback regarding this repository or survey!

🌟 If you find this resource helpful for your work, please consider citing our [research](#citation).


---

<a name="readme-index"></a>

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


<p align="center">
<img src="assets/literature.png" width = "90%" alt="" align=center />
</p>

---

## Temporal — Execution & Scheduling

These methods act on **when** KV data is executed, computed, or scheduled to improve latency and throughput. We divide these methods into three categories: KV-centric scheduling, pipelining & overlapping, and hardware-aware execution.

### KV-Centric Scheduling


|Paper|Type|Code|
| -- | -- | -- |
| [![Publish](https://img.shields.io/badge/Conference-EMNLP_2025-blue)]() <br> TokenSelect: Efficient Long-Context Inference and Length Extrapolation for LLMs via Dynamic Token-Level KV Cache Selection [[Link](https://arxiv.org/pdf/2411.02886)] <br> *Wei Wu, Zhuoshi Pan, Chao Wang, Liyi Chen, Yunchu Bai, Tianfu Wang, Kun Fu, Zheng Wang, Hui Xiong* |  |[![stars](https://img.shields.io/github/stars/pzs19/TokenSelect?style=social)](https://github.com/pzs19/TokenSelect) <br> ![](https://img.shields.io/github/last-commit/pzs19/TokenSelect?color=green) <br> [TokenSelect](https://github.com/pzs19/TokenSelect)|
| [![Publish](https://img.shields.io/badge/Conference-ACL_2025-blue)]() <br> RefreshKV: Updating Small KV Cache During Long-form Generation [[Link](https://aclanthology.org/2025.acl-long.1211.pdf)] <br> *Fangyuan Xu, Tanya Goyal,  Eunsol Choi*|  |[![stars](https://img.shields.io/github/stars/carriex/refreshkv?style=social)](https://github.com/carriex/refreshkv) <br> ![](https://img.shields.io/github/last-commit/carriex/refreshkv?color=green) <br> [RefreshKV](https://github.com/carriex/refreshkv)|
| [![Publish](https://img.shields.io/badge/Conference-MLSys_2025-cyan)]() ![Award](https://img.shields.io/badge/Outstanding%20Paper%20Award-gold?logo=star&logoColor=white) <br> FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving [[Link](https://openreview.net/pdf?id=RXPofAsL8F)] | Also belongs to allocation & reuse (structural) | [![stars](https://img.shields.io/github/stars/flashinfer-ai/flashinfer?style=social)](https://github.com/flashinfer-ai/flashinfer) <br> ![](https://img.shields.io/github/last-commit/flashinfer-ai/flashinfer?color=green) <br> [FlashInfer](https://github.com/flashinfer-ai/flashinfer) 🌟 |
| [![Publish](https://img.shields.io/badge/Conference-FAST_2025-cyan)]() ![Award](https://img.shields.io/badge/Best%20Paper%20Award-gold?logo=star&logoColor=white) <br> Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot [[Link](https://www.usenix.org/system/files/fast25-qin.pdf)] <br> *Ruoyu Qin, Zheming Li, Weiran He, Jialei Cui, Feng Ren, Mingxing Zhang, Yongwei Wu, Weimin Zheng, Xinran Xu* | Also belongs to HW-aware execution | [![stars](https://img.shields.io/github/stars/kvcache-ai/Mooncake?style=social)](https://github.com/kvcache-ai/Mooncake) <br> ![](https://img.shields.io/github/last-commit/kvcache-ai/Mooncake?color=green) <br>[Mooncake](https://github.com/kvcache-ai/Mooncake) 🌟 |
| [![Publish](https://img.shields.io/badge/Conference-NeurIPS_2024-blue)]() <br> Loki: Low-rank Keys for Efficient Sparse Attention [[Link](https://proceedings.neurips.cc/paper_files/paper/2024/file/1e027da6bec9ceb2ec37951ceeccae93-Paper-Conference.pdf)] <br> *Prajwal Singhania, Siddharth Singh, Shwai He, Soheil Feizi, Abhinav Bhatele* |  |[![stars](https://img.shields.io/github/stars/hpcgroup/loki?style=social)](https://github.com/hpcgroup/loki) <br> ![](https://img.shields.io/github/last-commit/hpcgroup/loki?color=green) <br> [Loki](https://github.com/hpcgroup/loki)  |
| [![Publish](https://img.shields.io/badge/Conference-NeurIPS_2024-blue)]() <br> SGLang: Efficient Execution of Structured Language Model Programs [[Link](https://proceedings.neurips.cc/paper_files/paper/2024/file/724be4472168f31ba1c9ac630f15dec8-Paper-Conference.pdf)] <br> *Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, Ying Sheng* | Also belongs to allocation & reuse (structural) |[![stars](https://img.shields.io/github/stars/sgl-project/sglang?style=social)](https://github.com/sgl-project/sglang) <br> ![](https://img.shields.io/github/last-commit/sgl-project/sglang?color=green) <br> [SGLang](https://github.com/sgl-project/sglang) 🌟 |
| [![Publish](https://img.shields.io/badge/Conference-SOSP_2024-cyan)]() <br> LoongServe: Efficiently Serving Long-Context Large Language Models with Elastic Sequence Parallelism [[Link](https://arxiv.org/pdf/2404.09526)] <br> *Bingyang Wu, Shengyu Liu, Yinmin Zhong, Peng Sun, Xuanzhe Liu, Xin Jin* |  |[![stars](https://img.shields.io/github/stars/LoongServe/LoongServe?style=social)](https://github.com/LoongServe/LoongServe) <br> ![](https://img.shields.io/github/last-commit/LoongServe/LoongServe?color=green) <br> [LoongServe](https://github.com/LoongServe/LoongServe) |
| Fast Inference for Augmented Large Language Models [[Link](https://arxiv.org/pdf/2410.18248)] <br> *Rana Shahout, Cong Liang, Shiji Xin, Qianru Lao, Yong Cui, Minlan Yu, Michael Mitzenmacher*|  | |
| LayerKV: Optimizing Large Language Model Serving with Layer-wise KV Cache Management [[Link](https://arxiv.org/pdf/2410.00428)] <br> *Yi Xiong, Hao Wu, Changxu Shao, Ziqing Wang, Rui Zhang, Yuhong Guo, Junping Zhao, Ke Zhang, Zhenxuan Pan*| Also belongs to memory hierarchy KV orchestration (spatial) | |
| [![Publish](https://img.shields.io/badge/Conference-ICML_2024-blue)]() <br> SparQ Attention: Bandwidth-Efficient LLM Inference [[Link](https://openreview.net/pdf?id=OS5dqxmmtl)] <br> *Luka Ribar, Ivan Chelombiev, Luke Hudlass-Galley, Charlie Blake, Carlo Luschi, Douglas Orr* |  | [![stars](https://img.shields.io/github/stars/graphcore-research/llm-inference-research?style=social)](https://github.com/graphcore-research/llm-inference-research/tree/2024-05-sparq) <br> ![](https://img.shields.io/github/last-commit/graphcore-research/llm-inference-research?color=green) <br> [SparQ Attention](https://github.com/graphcore-research/llm-inference-research/tree/2024-05-sparq)  |
| [![Publish](https://img.shields.io/badge/Conference-ICML_2024-blue)]() <br> QUEST: Query-Aware Sparsity for Efficient Long-Context LLM Inference [[Link](https://openreview.net/pdf?id=KzACYw0MTV)] <br> *Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, Song Han*|  |[![stars](https://img.shields.io/github/stars/mit-han-lab/quest?style=social)](https://github.com/mit-han-lab/quest) <br> ![](https://img.shields.io/github/last-commit/mit-han-lab/quest?color=green) <br> [Quest](https://github.com/mit-han-lab/quest)  |
| [![Publish](https://img.shields.io/badge/Conference-ICML_2024-blue)]() <br> MuxServe: Flexible Spatial-Temporal Multiplexing for Multiple LLM Serving [[Link](https://openreview.net/pdf?id=R0SoZvqXyQ)] <br> *Jiangfei Duan, Runyu Lu, Haojie Duanmu, Xiuhong Li, Xingcheng Zhang, Dahua Lin, Ion Stoica, Hao Zhang* | Also belongs to HW-aware execution |[![stars](https://img.shields.io/github/stars/hao-ai-lab/MuxServe?style=social)](https://github.com/hao-ai-lab/MuxServe) <br> ![](https://img.shields.io/github/last-commit/hao-ai-lab/MuxServe?color=green) <br> [MuxServe](https://github.com/hao-ai-lab/MuxServe)  |
| [![Publish](https://img.shields.io/badge/Conference-ICLR_2024-blue)]() <br> Preble: Efficient Distributed Prompt Scheduling for LLM Serving [[Link](https://openreview.net/pdf?id=meKEKDhdnx)] <br> *Vikranth Srivatsa, Zijian He, Reyna Abhyankar, Dongming Li, Yiying Zhang* |  |[![stars](https://img.shields.io/github/stars/WukLab/preble?style=social)](https://github.com/WukLab/preble) <br> ![](https://img.shields.io/github/last-commit/WukLab/preble?color=green) <br> [Preble](https://github.com/WukLab/preble) |
| Inference without interference: Disaggregate LLM inference for mixed downstream workloads [[Link](https://arxiv.org/pdf/2401.11181)] <br> *Cunchen Hu, Heyang Huang, Liangliang Xu, Xusheng Chen, Jiang Xu, Shuang Chen, Hao Feng, Chenxi Wang, Sa Wang, Yungang Bao, Ninghui Sun, Yizhou Shan* | Also belongs to HW-aware execution | |



<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-index" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Index ↑
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
    <a href="#readme-index" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Index ↑
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
    <a href="#readme-index" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Index ↑
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
| 2024 | FastSwitch: Optimizing Context Switching Efficiency in Fairness-aware Large Language Model Serving [[Link](https://arxiv.org/pdf/2411.18424)] | Also belongs to allocation & reuse (structural) | arXiv || 
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
    <a href="#readme-index" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Index ↑
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
    <a href="#readme-index" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Index ↑
    </a>
</p>

---

## Structural — Representation & Retention

These methods target **how** KV data is represented and maintained for memory efficiency. We divide these methods into two categories: KV cache compression, and KV cache retention management.

### KV Cache Compression
#### Quantization


|Year|Paper|Type|Venue|Code|
| -- | -- | -- | -- | -- |
| 2025 | Accurate KV Cache Quantization with Outlier Tokens Tracing [[Link](https://aclanthology.org/2025.acl-long.631.pdf)] |  | ACL |[OTT](https://github.com/yisunlp/OTT) [![stars](https://img.shields.io/github/stars/yisunlp/OTT?style=social)](https://github.com/yisunlp/OTT) |
| 2025 | CommVQ: Commutative Vector Quantization for KV Cache Compression [[Link](https://openreview.net/pdf?id=sbbyCB39HN)] |  | ICML |[CommVQ](https://github.com/UMass-Embodied-AGI/CommVQ) [![stars](https://img.shields.io/github/stars/UMass-Embodied-AGI/CommVQ?style=social)](https://github.com/UMass-Embodied-AGI/CommVQ) |
| 2025 | NSNQuant: A Double Normalization Approach for Calibration-Free Low-Bit Vector Quantization of KV Cache [[Link](https://arxiv.org/pdf/2505.18231)] |  | ICML | |
| 2025 | QServe: W4A8KV4 quantization and system co- design for efficient LLM serving [[Link](https://openreview.net/pdf/1ec600eaf0c56573a4d7a7818181657962d03d8f.pdf)] |  | MLSys |[OmniServe](https://github.com/mit-han-lab/omniserve) [![stars](https://img.shields.io/github/stars/mit-han-lab/omniserve?style=social)](https://github.com/UMass-Embodied-AGI/CommVQ) |
| 2025 | SQuat: Subspace-orthogonal KV cache quantization [[Link](https://arxiv.org/pdf/2503.24358)] |  | arXiv |[SQuat](https://github.com/Red-Hat-AI-Innovation-Team/SQuat) [![stars](https://img.shields.io/github/stars/Red-Hat-AI-Innovation-Team/SQuat?style=social)](https://github.com/Red-Hat-AI-Innovation-Team/SQuat) |
| 2025 | VQ-LLM: High-performance Code Generation for Vector Quantization Augmented LLM Inference [[Link](https://arxiv.org/pdf/2503.02236)] |  | HPCA | |
| 2025 | QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead [[Link](https://arxiv.org/pdf/2406.03482)] |  | AAAI |[QJL](https://github.com/amirzandieh/QJL) [![stars](https://img.shields.io/github/stars/amirzandieh/QJL?style=social)](https://github.com/amirzandieh/QJL) |
| 2024 | ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification [[Link](https://proceedings.neurips.cc/paper_files/paper/2024/file/7e57131fdeb815764434b65162c88895-Paper-Conference.pdf)] |  | NeurIPS |[ZipCache](https://github.com/ThisisBillhe/ZipCache) [![stars](https://img.shields.io/github/stars/ThisisBillhe/ZipCache?style=social)](https://github.com/ThisisBillhe/ZipCache) |
| 2024 | KV Cache is 1 Bit Per Channel: Efficient Large Language Model Inference with Coupled Quantization [[Link](https://openreview.net/pdf?id=pNnvzQsS4P)] |  | NeurIPS | |
| 2024 | KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization [[Link](https://openreview.net/pdf?id=0LXotew9Du)] |  | NeurIPS |[KVQuant](https://github.com/SqueezeAILab/KVQuant) [![stars](https://img.shields.io/github/stars/SqueezeAILab/KVQuant?style=social)](https://github.com/SqueezeAILab/KVQuant) |
| 2024 | SKVQ: Sliding-window Key and Value Cache Quantization for Large Language Models [[Link](https://openreview.net/pdf?id=nI6JyFSnyV)] |  | COLM |[SKVQ](https://github.com/cat538/SKVQ) [![stars](https://img.shields.io/github/stars/cat538/SKVQ?style=social)](https://github.com/cat538/SKVQ) |
| 2024 | GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM [[Link](https://arxiv.org/pdf/2403.05527)] |  | arXiv |[GEAR](https://github.com/opengear-project/GEAR) [![stars](https://img.shields.io/github/stars/opengear-project/GEAR?style=social)](https://github.com/opengear-project/GEAR) |
| 2024 | Unlocking Data-free Low-bit Quantization with Matrix Decomposition for KV Cache Compression [[Link](https://aclanthology.org/2024.acl-long.133.pdf)] |  | ACL |[DecoQuant](https://github.com/lpyhdzx/DecoQuant_code) [![stars](https://img.shields.io/github/stars/lpyhdzx/DecoQuant_code?style=social)](https://github.com/lpyhdzx/DecoQuant_code) |
| 2024 | CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving [[Link](https://dl.acm.org/doi/pdf/10.1145/3651890.3672274)] | Also belongs to compute device KV orchestration (spatial) | SIGCOMM |[CacheGen](https://github.com/UChi-JCL/CacheGen) [![stars](https://img.shields.io/github/stars/UChi-JCL/CacheGen?style=social)](https://github.com/UChi-JCL/CacheGen) |
| 2024 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache [[Link](https://openreview.net/pdf?id=L057s2Rq8O)] |  | ICML |[KIVI](https://github.com/jy-yuan/KIVI) [![stars](https://img.shields.io/github/stars/jy-yuan/KIVI?style=social)](https://github.com/jy-yuan/KIVI) |
| 2024 | Atom: Low-Bit Quantization for Efficient and Accurate LLM Serving [[Link](https://proceedings.mlsys.org/paper_files/paper/2024/file/5edb57c05c81d04beb716ef1d542fe9e-Paper-Conference.pdf)] |  | MLSys |[Atom](https://github.com/efeslab/Atom) [![stars](https://img.shields.io/github/stars/efeslab/Atom?style=social)](https://github.com/efeslab/Atom) |
| 2024 | QAQ: Quality Adaptive Quantization for LLM KV Cache [[Link](https://arxiv.org/pdf/2403.04643)] |  | arXiv |[QAQ](https://github.com/ClubieDong/QAQ-KVCacheQuantization) [![stars](https://img.shields.io/github/stars/ClubieDong/QAQ-KVCacheQuantization?style=social)](https://github.com/ClubieDong/QAQ-KVCacheQuantization) |
| 2024 | No Token Left Behind: Reliable KV Cache Compression via Importance-Aware Mixed Precision Quantization [[Link](https://arxiv.org/pdf/2402.18096)] |  | arXiv ||
| 2024 | WKVQuant: Quantizing Weight and Key/Value Cache for Large Language Models Gains More [[Link](https://arxiv.org/pdf/2402.12065)] |  | arXiv ||
| 2023 | FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU [[Link](https://openreview.net/pdf?id=RRntzKrBTp)] | Also belongs to memory hierarchy KV orchestration | ICML |[FlexLLMGen](https://github.com/FMInference/FlexLLMGen) 🌟 [![stars](https://img.shields.io/github/stars/FMInference/FlexLLMGen?style=social)](https://github.com/FMInference/FlexLLMGen) |
| 2023 | SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models [[Link](https://proceedings.mlr.press/v202/xiao23c/xiao23c.pdf)] | | ICML |[SmoothQuant](https://github.com/mit-han-lab/smoothquant) 🌟 [![stars](https://img.shields.io/github/stars/mit-han-lab/smoothquant?style=social)](https://github.com/mit-han-lab/smoothquant) |


#### Low-rank Approximation


|Year|Paper|Type|Venue|Code|
| -- | -- | -- | -- | -- |
| 2025 | ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference [[Link](https://openreview.net/pdf?id=oa7MYAO6h6)] | Also belongs memory hierarchy KV orchestration (spatial) | ICML 💡 **Spotlight** |[ShadowKV](https://github.com/ByteDance-Seed/ShadowKV) [![stars](https://img.shields.io/github/stars/ByteDance-Seed/ShadowKV?style=social)](https://github.com/ByteDance-Seed/ShadowKV) |
| 2025 | FDC: Fast KV Dimensionality Compression for Efficient LLM Inference [[Link](https://arxiv.org/pdf/2408.04107)] | | arXiv | |
| 2025 | ReCalKV: Low-Rank KV Cache Compression via Head Reordering and Offline Calibration [[Link](https://arxiv.org/pdf/2505.24357)] |  | arXiv |[ReCalKV](https://github.com/XIANGLONGYAN/ReCalKV) [![stars](https://img.shields.io/github/stars/XIANGLONGYAN/ReCalKV?style=social)](https://github.com/XIANGLONGYAN/ReCalKV) |
| 2025 | MatryoshkaKV: Adaptive KV Compression via Trainable Orthogonal Projection [[Link](https://openreview.net/pdf?id=BQwsRy1h3U)] |  | ICLR |[MatryoshkaKV-cache](https://github.com/The-kamisato-Sii/MatryoshkaKV-cache) [![stars](https://img.shields.io/github/stars/The-kamisato-Sii/MatryoshkaKV-cache?style=social)](https://github.com/The-kamisato-Sii/MatryoshkaKV-cache) |
| 2025 | xKV: Cross-Layer SVD for KV-Cache Compression [[Link](https://arxiv.org/pdf/2503.18893)] |  | arXiv |[xKV](https://github.com/abdelfattah-lab/xKV) [![stars](https://img.shields.io/github/stars/abdelfattah-lab/xKV?style=social)](https://github.com/abdelfattah-lab/xKV) |
| 2025 | Palu: KV-Cache Compression with Low-Rank Projection [[Link](https://openreview.net/pdf?id=LWMS4pk2vK)] |  | ICLR |[Palu](https://github.com/shadowpa0327/Palu) [![stars](https://img.shields.io/github/stars/shadowpa0327/Palu?style=social)](https://github.com/shadowpa0327/Palu) |
| 2024 | LoRC: Low-Rank Compression for LLMs KV Cache with a Progressive Compression Strategy [[Link](https://arxiv.org/pdf/2410.03111)] |  | arXiv | |
| 2025 | Get More with LESS: Synthesizing Recurrence with KV Cache Compression for Efficient LLM Inference [[Link](https://openreview.net/pdf?id=uhHDhVKFMW)] |  | ICML |[LESS](https://github.com/hdong920/LESS) [![stars](https://img.shields.io/github/stars/hdong920/LESS?style=social)](https://github.com/hdong920/LESS) |
| 2024 | Effectively Compress KV Heads for LLM [[Link](https://arxiv.org/pdf/2406.07056)] |  | arXiv | |



#### Structural Compression

Still work in progress.



<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-index" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Index ↑
    </a>
</p>

### KV Cache Retention Management
#### Allocation & Reuse

|Year|Paper|Type|Venue|Code|
| -- | -- | -- | -- | -- |
| 2025 | FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving [[Link](https://openreview.net/pdf?id=RXPofAsL8F)] | Also belongs to KV-centric scheduling (temporal) | MLSys 🏆 **Outstanding Paper Award** |[FlashInfer](https://github.com/flashinfer-ai/flashinfer) 🌟 [![stars](https://img.shields.io/github/stars/flashinfer-ai/flashinfer?style=social)](https://github.com/flashinfer-ai/flashinfer) |
| 2025 | Unifying KV Cache Compression for Large Language Models with LeanKV [[Link](https://arxiv.org/pdf/2412.03131v2)] |  | arXiv | |
| 2025 | vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention [[Link](https://dl.acm.org/doi/pdf/10.1145/3669940.3707256)] |  | ASPLOS | [vAttention](https://github.com/microsoft/vattention) [![stars](https://img.shields.io/github/stars/microsoft/vattention?style=social)](https://github.com/microsoft/vattention) |
| 2025 | MemServe: Context Caching for Disaggregated LLM Serving with Elastic Memory Pool [[Link](https://arxiv.org/pdf/2406.17565)] |  | arXiv |  |
| 2024 | SGLang: Efficient Execution of Structured Language Model Programs [[Link](https://proceedings.neurips.cc/paper_files/paper/2024/file/724be4472168f31ba1c9ac630f15dec8-Paper-Conference.pdf)] | Also belongs to KV-centric scheduling (temporal) | NeurIPS |[SGLang](https://github.com/sgl-project/sglang) 🌟 [![stars](https://img.shields.io/github/stars/sgl-project/sglang?style=social)](https://github.com/sgl-project/sglang) |
| 2024 | FastSwitch: Optimizing Context Switching Efficiency in Fairness-aware Large Language Model Serving [[Link](https://arxiv.org/pdf/2411.18424)] | Also belongs to memory hierarchy KV orchestration (spatial) | arXiv || 
| 2024 | ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition [[Link](https://aclanthology.org/2024.acl-long.623.pdf)] |  | ACL | [Chunk Attention](https://github.com/microsoft/chunk-attention) [![stars](https://img.shields.io/github/stars/microsoft/chunk-attention?style=social)](https://github.com/microsoft/chunk-attention) |
| 2024 | vTensor: Flexible Virtual Tensor Management for Efficient LLM Serving [[Link](https://arxiv.org/pdf/2407.15309)] |  | aXiv |  |
| 2024 | LazyLLM: Dynamic Token Pruning for Efficient Long Context LLM Inference [[Link](https://arxiv.org/pdf/2407.14057)] |  | arXiv |  |
| 2024 | Prompt Cache: Modular Attention Reuse for Low-Latency Inference [[Link](https://proceedings.mlsys.org/paper_files/paper/2024/file/a66caa1703fe34705a4368c3014c1966-Paper-Conference.pdf)] |  | MLSys | [Prompt Cache](https://github.com/yale-sys/prompt-cache) [![stars](https://img.shields.io/github/stars/yale-sys/prompt-cache?style=social)](https://github.com/yale-sys/prompt-cache) |
| 2023 | Efficient Memory Management for Large Language Model Serving with PagedAttention [[Link](https://dl.acm.org/doi/pdf/10.1145/3600006.3613165)] |  | SOSP | [vllm](https://github.com/vllm-project/vllm) 🌟 [![stars](https://img.shields.io/github/stars/vllm-project/vllm?style=social)](https://github.com/vllm-project/vllm) |







#### Eviction

Still work in progress.



<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-index" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Index ↑
    </a>
</p>

---

## Cross-behavior Synergies

Still work in progress.

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-index" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Index ↑
    </a>
</p>

---


## Behavior-objective Synergies

Still work in progress.


<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-index" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Index ↑
    </a>
</p>





## Citation

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



<!-- ### Contributors

<a href="https://github.com/atfortes/Awesome-KV-Cache-Optimization/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=atfortes/Awesome-KV-Cache-Optimization" />
</a> -->