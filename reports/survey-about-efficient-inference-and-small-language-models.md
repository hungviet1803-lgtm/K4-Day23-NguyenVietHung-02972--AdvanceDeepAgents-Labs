# Efficient Inference and Small Language Models: Methods, Measurement, and 2024–2026 Trends

## TL;DR
- Efficient inference is best understood as reducing compute, memory traffic, and decode-time serial work so that latency, throughput, power, and storage improve under real deployment constraints. Roofline-style analysis and prefill/decode separation are useful ways to reason about where bottlenecks sit. [1][2][3]
- Small language models are now framed less by a single parameter cutoff and more by context: they target on-device, edge, and bandwidth-constrained settings while trying to preserve adaptability and task quality. Recent surveys also narrow practical SLM studies to roughly 100M–5B decoder-only transformers. [4][5][6]
- The major technique families attack different bottlenecks: quantization lowers precision and memory cost, distillation trains a cheaper student, speculative decoding reduces decode-time seriality, and KV-cache paging removes serving fragmentation. [7][8][9][10]
- Evaluation has split into two layers: task quality benchmarks such as MMLU, GSM8K, and TruthfulQA, and systems metrics such as prefill latency, decode latency, throughput, memory footprint, and sometimes energy. These metrics can disagree, especially under batching, quantization, and serving optimizations. [5][11][12][13][14]
- Since 2024, the center of gravity has shifted toward deployment-aware co-design: newer work emphasizes simultaneous inference, KV-cache compression, speculative decoding inside serving stacks, and device-aware benchmarking for edge/mobile use. [15][16][17][5][18][19][20]

## Background
Efficient inference for language models is the problem of serving models with lower latency, higher throughput, and lower memory or energy cost without sacrificing too much output quality. Recent surveys describe the main pressure points as large model size, attention cost, and autoregressive decoding, which make inference expensive in latency, throughput, power, and storage terms. [1][2]

A useful mental model is to separate the prefill stage from the decode stage and ask whether the workload is memory-bound or compute-bound. Roofline-style analysis does exactly that by relating operational intensity to memory bandwidth and compute ceilings; this is especially relevant because LLM inference often spends heavily on memory traffic rather than pure arithmetic. [2][3]

Small language models are usually not defined by one fixed parameter threshold. Instead, surveys frame them relative to context and deployment constraints such as training or inference hardware, bandwidth, generation time, storage, and energy. Recent measurement-oriented work often studies decoder-only transformer SLMs in the 100M–5B range, because that is the regime where on-device and edge deployment become practical. [4][5][6]

## Main technique families
### Quantization and compression
Quantization is one of the most direct ways to reduce inference cost: lower-precision weights and activations reduce memory footprint and can speed matrix multiplications. SmoothQuant is a representative post-training method that migrates quantization difficulty from activations to weights through an equivalent transformation; it reports up to 1.56× speedup and 2× memory reduction with negligible accuracy loss. [8]

For long-context deployment, KV-cache quantization has become important because cache size can dominate serving memory. Recent edge-oriented work argues that even sub-billion-parameter models hit unavoidable long-context memory bottlenecks, motivating lower-bit key/value caching strategies and attention-aware quantization schemes. [18]

### Distillation and compact students
Distillation changes the model itself: a smaller student is trained to imitate a larger teacher, so inference becomes cheaper because the deployed model is smaller. DistilBERT is the classic reference point: it cuts BERT size by 40%, is 60% faster, and retains 97% of BERT’s language understanding capabilities. The key tradeoff is that the efficiency gain is larger than with purely runtime tricks, but it requires extra training and can reduce fidelity or adaptability. [10]

### Speculative and simultaneous decoding
Speculative decoding attacks decode-time seriality. Instead of generating every token with the large model in sequence, a small draft model proposes multiple tokens and the target model verifies them, yielding exact outputs faster. The original method reported 2×–3× acceleration on T5-XXL without retraining or architecture changes. [7][21]

Recent serving systems extend this idea in deployment-aware ways. vLLM’s documentation and blog emphasize that speculative decoding helps most at low QPS or memory-bound workloads, and that newer variants such as dynamic speculative decoding and adaptive verification are designed to improve latency further. A related 2024 line of work, simultaneous inference, enables predictions with incomplete prompts to reduce response latency while preserving or improving accuracy. [16][19][20]

### KV-cache management and serving systems
For decoder-only models, KV cache management is central because long contexts and concurrent requests create fragmentation and memory waste. PagedAttention partitions the cache into blocks and maps logical to non-contiguous physical blocks, enabling flexible sharing and near-zero waste in KV cache memory; vLLM reports 2×–4× throughput gains versus FasterTransformer and Orca without affecting model accuracy. [22][9]

This family matters because serving cost is often dominated by memory behavior rather than raw FLOPs. In practice, paged cache layouts, continuous batching, and block sharing are now part of the baseline stack for high-throughput inference. [22][9][20]

## Evaluation and benchmarks
Evaluation splits into quality metrics and systems metrics. For quality, common academic leaderboards use multiple-choice or loglikelihood-based tasks such as ARC, HellaSwag, MMLU, TruthfulQA, Winogrande, and GSM8K, while the lm-evaluation-harness exposes metrics like acc, acc_norm, perplexity, and weighted_perplexity. These are useful for comparing reasoning and knowledge retention, but they mostly ignore deployment behavior. [11][12]

Newer SLM studies combine those task suites with runtime metrics. Measurement-focused work reports first-token latency for prefill, latency per token for decode, runtime memory usage, and sometimes energy consumption; edge-focused benchmarks also standardize prompt length, token generation length, maximum context length, and quantization settings to make runtime comparisons more reproducible. [5][5][23]

Edge and mobile benchmarks try to bring deployment realism into evaluation. Mobile-MMLU, for example, emphasizes latency, energy, memory, and response quality in mobile settings, while MMLU-Pro increases difficulty and answer-choice count to reduce benchmark saturation and prompt sensitivity. The broader lesson is that a model can score well on static accuracy benchmarks yet still be unattractive in real serving because of memory, latency, or energy cost. [13][24][14]

## Trends and open problems
Since 2024, efficient inference has become more deployment-aware and more system co-designed. Surveys now explicitly highlight model-hardware co-design, vocabulary and KV-cache compression, and routing across models or tasks as major opportunities rather than niche optimizations. [1][5][6]

A clear trend is that the strongest recent systems combine multiple tricks: speculative decoding within serving stacks, continuous batching, incomplete-prompt or simultaneous inference, and KV-cache-aware memory management. These methods improve responsiveness, but they can become less attractive at higher QPS or under different batching regimes, so “best” is increasingly workload-dependent rather than universal. [16][19][20]

For small language models, the picture is more optimistic than a year ago: recent benchmarking suggests state-of-the-art SLMs can outperform 7B models on general tasks, which makes them credible deployable models rather than only compressed fallback options. At the same time, surveys still identify in-context learning, long-context memory, and efficient routing as unresolved weaknesses. [4][5][6][17]

The main open problems are therefore less about whether efficiency tricks work in isolation and more about how they compose. We still lack a universally accepted benchmark that jointly captures quality, latency, memory, energy, and serving behavior across hardware targets; we also lack clear guidance for when to use a smaller model, a faster decode algorithm, or a systems-level cache optimization. [5][23][13][14][15][18][20]

## References
[1] A Survey on Efficient Inference for Large Language Models. arxiv. https://arxiv.org/abs/2404.14294 (2024-04-22)
[2] LLM Inference Unveiled: Survey and Roofline Model Insights. arxiv. https://arxiv.org/abs/2402.16363 (2024-02-26)
[3] Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures. web. https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/Archive/EECS-2008-134.pdf (n.d.)
[4] A Survey of Small Language Models. web. https://arxiv.org/html/2410.20011v1 (2024-10-25)
[5] Small language models: Survey, measurements, and insights. arxiv. https://arxiv.org/abs/2409.15790 (2025-02-26)
[6] A Survey on Small Language Models. web. https://aclanthology.org/2025.ranlp-1.93/ (n.d.)
[7] Fast Inference from Transformers via Speculative Decoding. arxiv. https://arxiv.org/abs/2211.17192 (2023-05-18)
[8] SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. arxiv. https://arxiv.org/abs/2211.10438 (2024-03-29)
[9] Efficient Memory Management for Large Language Model Serving with PagedAttention. arxiv. https://arxiv.org/abs/2309.06180 (2023-09-12)
[10] DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arxiv. https://arxiv.org/abs/1910.01108 (2019-10-02)
[11] Open LLM Leaderboard v1 - Hugging Face. web. https://huggingface.co/docs/leaderboards/en/open_llm_leaderboard/archive (n.d.)
[12] Scoring & Metrics ¶. web. https://lm-evaluation-harness.readthedocs.io/writing_tasks/scoring_and_metrics/ (n.d.)
[13] Mobile-MMLU: A Mobile Intelligence Language Understanding Benchmark. web. https://arxiv.org/abs/2503.20786 (n.d.)
[14] Chatbot Arena: Benchmarking LLMs in the Wild with Elo Ratings. web. https://www.lmsys.org/blog/2023-05-03-arena/ (2023-05-03)
[15] A Survey on Efficient Inference for Large Language Models. hf-search. https://huggingface.co/papers/2404.14294 (2024-04-22)
[16] LiveMind: Low-latency Large Language Models with Simultaneous Inference. hf-search. https://huggingface.co/papers/2406.14319 (2024-06-20)
[17] Demystifying Small Language Models for Edge Deployment. web. https://aclanthology.org/2025.acl-long.718/ (n.d.)
[18] Subkv: Quantizing Long Context KV Cache for Sub‐Billion Parameter Language Models on Edge Devices. web. https://oar.a-star.edu.sg/communities-collections/articles/22330 (2025-04-02)
[19] Speculative Decoding - vLLM Documentation. web. https://docs.vllm.ai/en/stable/features/speculative_decoding/ (n.d.)
[20] How Speculative Decoding Boosts vLLM Performance by up to .... web. https://vllm.ai/blog/2024-10-17-spec-decode (2024-10-17)
[21] Fast Inference from Transformers via Speculative Decoding. web. https://proceedings.mlr.press/v202/leviathan23a.html (2023-07-03)
[22] Paged Attention - vLLM Documentation. web. https://docs.vllm.ai/en/latest/design/paged_attention/ (n.d.)
[23] Demystifying Small Language Models for Edge Deployment. web. https://aclanthology.org/anthology-files/anthology-files/pdf/acl/2025.acl-long.718.pdf (n.d.)
[24] MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark. web. https://arxiv.org/html/2406.01574 (2024-11-06)
