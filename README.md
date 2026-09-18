<h1 align="center">Gian Luigi D'Alessandro</h1>

<p align="center">
  <b>Founding AI Research Engineer</b> · RL for LLM agents · training infra · GPU kernels
</p>

<p align="center">
  I build the tooling that lets LLM agents improve from their own experience.<br/>
  Ex-CTO &amp; co-founder @ Unakin. Particle physics PhD — Oxford / Royal Holloway, based at CERN.
</p>

<p align="center">
  <a href="https://linkedin.com/in/giandalessandro">LinkedIn</a> &nbsp;·&nbsp;
  <a href="mailto:gianluigidalessandro23@gmail.com">Email</a> &nbsp;·&nbsp;
  <a href="https://github.com/gian-g3dai">GitHub</a>
</p>

---

```python
>>> from gian import profile
>>> profile.now()
Role(title="Founding AI Research Engineer", company="stealth", loc="London, UK")
>>> profile.thesis
'agents should learn from their own execution traces, not from prompt tweaking'
```

### `whoami`

Research engineer working on **reinforcement learning for LLM agents** — I research the methods,
then ship the ones that survive contact with reality as a Python SDK. Composable primitives for
**rewards, rollouts, and evals**, PyTorch-style, so training an agent feels like writing code
rather than tuning prompts.

Before that I co-founded **Unakin** (CTO) and built **Sawyer**, an autonomous coding agent for
game development, from scratch: custom Llama 2/3 training, our own 8× A100 cluster, and a
multi-agent orchestration stack deployed to enterprise customers.

### `git log --author="gian" --oneline`

**Founding AI Research Engineer — Stealth** · `Aug 2026 → HEAD` · London

- Python SDK for training LLM agents with RL — one training loop, any agent plugged into it
- Researching RL methods for agents that learn from their own execution traces: GRPO-style policy
  optimisation, benchmarked against prompt-optimisation approaches like GEPA's Pareto search
- Composable primitives for rewards, rollouts and evals (`nn.Module` energy, but for agents)
- The eval harness that decides what ships — improvement on held-out tasks, not on training reward

**Co-founder & CTO — Unakin** · `Jan 2023 → Jul 2026` · London

- Designed and built **Sawyer**: an autonomous coding agent that plans, writes and iterates on game
  code inside Unity with full project context (code, scenes, assets)
- Trained custom **Llama 2/3** models in PyTorch — full fine-tuning, extended pre-training, LoRA;
  multi-stage pipelines on distributed infra (**FSDP / DDP**, custom LoRA-aware sharding), 8× A100
- Wrote a custom **Triton cross-entropy kernel**: ~50% memory cut vs. stock PyTorch → 2× larger batches
- Shipped multi-agent orchestration + real-time inference to enterprise customers
- Fine-tuned text-conditioned **diffusion models** for 3D generation (DreamBooth, LoRA); ~10× faster
  inference by swapping out the NeRF backbone at inference time
- Built and led a 6-person engineering team; owned technical strategy and architecture
- Raised **$1.5M pre-seed** (Betaworks, Greycroft, Entrepreneur First, Mercuri) + $250K Google for
  Startups; selected for NVIDIA Inception

**PhD, Particle Physics — John Adams Institute (Oxford / RHUL), at CERN**

- Led development of **Monte Carlo simulation** software and distributed data-processing pipelines
  for high-energy physics experiments
- Core contributor to [pyg4ometry](https://github.com/jairhul/pyg4ometry), an open-source geometry
  library used across HEP

### `focus`

`RL for LLM agents` &nbsp; `GRPO / policy optimisation` &nbsp; `agent evals & reward design` &nbsp; `post-training (SFT · LoRA · RLHF)` &nbsp; `distributed training (FSDP · DDP · TP)` &nbsp; `GPU kernels (Triton · CUDA)` &nbsp; `multi-agent orchestration`

### `stack`

![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white&style=flat-square)
![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?logo=pytorch&logoColor=white&style=flat-square)
![CUDA / Triton](https://img.shields.io/badge/-CUDA%20%2F%20Triton-76B900?logo=nvidia&logoColor=white&style=flat-square)
![Hugging Face](https://img.shields.io/badge/-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black&style=flat-square)
![C++](https://img.shields.io/badge/-C++-00599C?logo=cplusplus&logoColor=white&style=flat-square)
![C#](https://img.shields.io/badge/-C%23-239120?style=flat-square)
![Docker](https://img.shields.io/badge/-Docker-2496ED?logo=docker&logoColor=white&style=flat-square)
![Kubernetes](https://img.shields.io/badge/-Kubernetes-326CE5?logo=kubernetes&logoColor=white&style=flat-square)
![Google Cloud](https://img.shields.io/badge/-GCP-4285F4?logo=googlecloud&logoColor=white&style=flat-square)
![Linux](https://img.shields.io/badge/-Linux-FCC624?logo=linux&logoColor=black&style=flat-square)

### `selected work`

- **[LLM-Research](https://github.com/gian-g3dai/LLM-Research)** — Code Llama fine-tuning with PyTorch
  FSDP, CUDA-graph optimisation, and memory-efficient split cross-entropy
- **[Poseidon-Triton](https://github.com/gian-g3dai/Poseidon-Triton)** — end-to-end LLM inference
  serving with NVIDIA Triton + TensorRT-LLM
- **[pyg4ometry](https://github.com/jairhul/pyg4ometry)** — open-source geometry library for HEP
  simulation (contributor)
