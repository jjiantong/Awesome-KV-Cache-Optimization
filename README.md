<h1 align="center">Awesome KV Cache Optimization</h1>

<!-- <p align="center">
    <b> Curated collection of papers on system-aware, serving-time, KV-centric techniques.</b>
</p> -->

<p align="center">
    <img src="assets/awesome-cover.png" width="80%"  style="align:center;"/>
</p>

This repository is dedicated to recording papers on system-aware, serving-time, KV-centric techniques, which serves as supplementary materials for our survey paper:

> **Towards Efficient Large Language Model Serving: A Survey on System-Aware KV Cache Optimization**  
> 📄 [TechRxiv Preprint](https://doi.org/10.36227/techrxiv.176046306.66521015/v1) (DOI: 10.36227/techrxiv.176046306.66521015/v1)  
> 🧑‍💻👩‍💻 *[Jiantong Jiang](https://jjiantong.github.io/)<sup>1</sup>, [Peiyu Yang](https://ypeiyu.github.io/)<sup>1</sup>, [Rui Zhang](https://www.ruizhang.info/)<sup>2</sup>, and [Feng Liu](https://fengliu90.github.io/)<sup>1</sup>*  
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
    - [Disaggregated Inference](#disaggregated-inference)
    - [Compute Offloading](#compute-offloading)
- [Spatial — Placement \& Migration](#spatial--placement--migration)
  - [Memory Hierarchy KV Orchestration](#memory-hierarchy-kv-orchestration)
    - [Cross-device Memory Hierarchy](#cross-device-memory-hierarchy)
    - [Intra-GPU Memory Hierarchy](#intra-gpu-memory-hierarchy)
  - [Compute Device KV Orchestration](#compute-device-kv-orchestration)
- [Structural — Representation \& Retention](#structural--representation--retention)
  - [KV Cache Compression](#kv-cache-compression)
    - [Quantization](#quantization)
    - [Low-Rank Approximation](#low-rank-approximation)
    - [Structural Compression](#structural-compression)
  - [KV Cache Retention Management](#kv-cache-retention-management)
    - [Allocation \& Reuse](#allocation--reuse)
    - [Eviction](#eviction)
- [Cross-Behavior Synergies](#cross-behavior-synergies)
- [Citation](#citation)

---

## Temporal — Execution & Scheduling

These methods act on **when** KV data is executed, computed, or scheduled to improve latency and throughput.

### KV-Centric Scheduling

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>

### Pipelining & Overlapping

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>

### Hardware-Aware Execution

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

#### Low-Rank Approximation

#### Structural Compression

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>

### KV Cache Retention Management
#### Allocation & Reuse

#### Eviction

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>

---

## Cross-Behavior Synergies


<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>

---


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