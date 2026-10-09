# Survey of Video and Multimodal Generation

## TL;DR
- Since 2024, diffusion transformers have emerged as the leading modern backbone for video generation, with Sora-style systems operating on compressed spacetime patches and supporting variable duration and resolution; the main tradeoff is scalability versus compute and long-horizon coherence [1][2][3].
- The field is increasingly organized by conditioning and modality: text-to-video, image-to-video, video-to-video, audio- or pose-conditioned generation, and broader multimodal generation are now standard categories [4][5][6][7].
- Evaluation is shifting from single-score fidelity toward structured benchmark suites such as VBench, VBench-2.0, and VBench++, plus human-preference leaderboards and multimodal benchmarks like MMMG [8][9][10][11][12].
- Open problems remain concentrated in physics, object permanence, long-duration consistency, complex scene layout, and synchronized generation across video, audio, text, and 3D [2][5][13][14][15].

## Background
Video and multimodal generation refer to models that create coherent outputs across time and across modalities, especially video, audio, text, images, and 3D. Recent surveys no longer treat this as a single task family; instead, they organize the space by output modality, conditioning modality, and representation choice, such as 2D appearance, temporal dynamics, geometry, or 4D world simulation [13][6][16][14]. In this framing, video is the step from static appearance to appearance plus dynamics, while 3D and 4D generalize further to geometry and full spatiotemporal scene structure [13].

This matters now because the field has moved from short clips and narrowly controlled generation toward longer, higher-resolution, and more general world-simulator-like systems. OpenAI’s Sora announcement explicitly framed video generation models as world simulators that can produce a minute of high-definition video from text conditioning, while also acknowledging weaknesses in physics and long-range coherence [1]. Survey work from 2024–2026 echoes that the core research challenge is not only visual realism, but also temporal consistency, physical plausibility, semantic alignment, and controllability [4][5][14].

## Main model families
The clearest modern backbone family is diffusion-based generation. Surveys describe the field’s evolution from earlier 3D U-Nets and temporal attention layers to diffusion transformers that operate on compressed spacetime patches, often with causal or windowed attention and cascaded refinement or super-resolution stages [1][17][2][3]. The attraction of this family is strong output quality and flexibility over duration, aspect ratio, and resolution; the recurring downside is computational cost and the difficulty of preserving coherence over long videos [1][17][18][19].

Within diffusion-based systems, several architectural patterns recur. Sora is described as using a transformer over tokenized latent video patches, with a compressed latent space and a decoder back to pixel space [1][17]. Other systems such as W.A.L.T are summarized as using causal encoding, window attention, and cascading super-resolution, while VDT and Latte extend diffusion transformers with spatiotemporal attention variants and conditional injection modules [2][3]. The common trend is to replace purely convolutional video backbones with transformerized diffusion pipelines that scale more naturally with data and compute.

A second family is autoregressive or token-based generation. Surveys position these systems as operating over discrete video tokens and transformer decoders, which makes them attractive for unified multimodal or masked-token settings, but also introduces compression and discretization tradeoffs [3]. In the broader multimodal literature, these systems are increasingly discussed alongside joint image-text or image-audio generators rather than in isolation, suggesting that token-based generation is becoming part of a more general multimodal stack [16][20][21].

A third family is GAN/VAE-based generation, which remains important historically and in some specialized pipelines, but is no longer the dominant approach for general-purpose high-fidelity text-to-video systems [6][3]. Contemporary surveys still list it as one of the principal families, mainly as a baseline or a legacy regime against which diffusion-transformer systems are compared [6][3].

Finally, there is growing interest in control-conditioned and task-specific subfamilies. Human video generation is now organized around text-, audio-, and pose-conditioned variants, and unified-modal video generation systems such as UniVG explicitly target multiple conditioning modes through mechanisms like multi-condition cross attention and biased Gaussian noise [7][21]. This indicates that “video generation” is increasingly defined by the conditioning signal as much as by the backbone architecture.

## Evaluation, benchmarks, and results
Evaluation has become more structured because video quality is multidimensional. A common decomposition is visual fidelity, temporal consistency, semantic alignment, and physical consistency, with video-specific issues including motion direction, event ordering, and attribute transitions over time [5]. Traditional distributional measures such as FVD remain part of the discussion, but surveys now emphasize that they are insufficient on their own because they do not fully capture human preference, temporal causality, or world-model quality [5].

VBench is the central structured benchmark family. The original suite decomposes video generation into 16 dimensions and is designed to align with human perception [8]. VBench-2.0 extends this toward intrinsic faithfulness, evaluating five high-level dimensions—Human Fidelity, Controllability, Creativity, Physics, and Commonsense—across 18 fine-grained capabilities [9]. VBench++ further broadens coverage to both text-to-video and image-to-video, adds an adaptive image suite for fairer image-conditioned evaluation, and reports 32 new models on the leaderboard [10].

Benchmarking has also become more explicit about human preference. Artificial Analysis’ AA-Video-T2V leaderboard ranks models by Elo from pairwise human votes, and the current leaderboard example in the notes reports Wan 3.0 at 1156 Elo, Utopai X at 1149, and Dreamina Seedance 2.5 at 1143 [11]. A separate VBench leaderboard page reports total and sub-dimension scores, including HiDream-O1-Video with a total score of 89.74, UN Veo 3 with 85.06, and UN Open-Sora-2.0 with 84.34 [22]. These leaderboards show that the field now compares systems using both automated metrics and human preference aggregation.

The benchmark picture is broader than video alone. ChronoMagic-Bench focuses on text-to-time-lapse generation and evaluates metamorphic amplitude and temporal coherence [23]. For multimodal generation, MMMG provides a human-aligned benchmark spanning image, audio, and interleaved text modalities across 49 tasks and 937 instructions, with 94.3% agreement with human evaluation [12]. UEval is a newer benchmark for unified multimodal generation of image and text, using expert-curated questions and rubric-based scoring [24]. Together, these systems show a shift from generic quality scoring toward capability-specific, human-aligned evaluation.

## Trends and open problems
The dominant trend since 2024 is scale plus unification. Video generators are being pushed toward longer duration, higher resolution, and more general “world simulator” behavior, while multimodal generators increasingly aim to unify video, audio, image, and 3D generation within one system or one planning framework [13][14][1][25][20]. At the same time, model families are being reorganized around conditioning and control, with prompt planning, layout planning, temporal prompt generation, and other LLM-assisted mechanisms gaining prominence [20][26].

A second trend is efficiency engineering. While diffusion transformers are strong performers, their compute cost motivates acceleration methods such as adaptive caching and motion regularization for diffusion transformers [18], as well as training-free sparse attention exploiting redundancy patterns [19]. These methods point to a practical tension: the best quality systems are also the hardest to run efficiently, especially for high-resolution or long-video generation.

The open problems are well aligned across surveys. Long-horizon temporal consistency, object permanence, physics plausibility, and coherent multi-object interactions remain unresolved [13][5][14][1]. Complex scenes and rational layout are still difficult, especially when several entities must move, interact, and remain identifiable over time [14][15]. Another unresolved issue is synchronized multimodal generation: truly aligned video, audio, and text generation remains a frontier rather than a solved capability [6][16][26].

There is also an ongoing debate about what kind of system design will matter most. One camp favors ever-larger unified generators that behave like world models; another emphasizes modular pipelines that separate planning, conditioning, generation, and refinement [5][16][20][21]. The evaluation gap makes this debate harder to settle, because current metrics still struggle to capture realism, causality, and world-model quality in a way that is fully aligned with human judgment [5][9][12].

Overall, the last two years have not produced a single new paradigm so much as a convergence: diffusion transformers dominate high-quality video generation, multimodal generation is becoming more unified, and benchmarking is moving toward capability-based and human-aligned evaluation. The field’s central unsolved question is whether the next step is simply larger and more efficient generators, or a qualitatively better way to represent and evaluate spatiotemporal world understanding [13][5][14][9].

## References
[1] Video generation models as world simulators. web. https://openai.com/index/video-generation-models-as-world-simulators/ (2024-02-15)
[2] A Survey on Video Diffusion Models | ACM Computing Surveys. web. https://dl.acm.org/doi/full/10.1145/3696415 (2024-11-07)
[3] From Sora What We Can See: A Survey of Text-to-Video Generation. web. https://arxiv.org/abs/2405.10674v1 (2024-05-17)
[4] Video diffusion generation: comprehensive review and open problems. web. https://link.springer.com/article/10.1007/s10462-025-11331-6 (2025-08-20)
[5] Generative AI Video Evaluation: Survey of Metrics, Benchmarks, and Trustworthiness. web. https://openaccess.thecvf.com/content/CVPR2026W/VGBE/papers/Safavigerdini_Generative_AI_Video_Evaluation_Survey_of_Metrics_Benchmarks_and_Trustworthiness_CVPRW_2026_paper.pdf (n.d.)
[6] AI-Generated Content (AIGC) for Various Data Modalities: A Survey. web. https://dl.acm.org/doi/10.1145/3728633 (2025-05-06)
[7] A Comprehensive Survey on Human Video Generation: Challenges, Methods, and Insights. web. https://github.com/wentaoL86/Awesome-Human-Video-Generation/blob/d97da607/README.md (n.d.)
[8] VBench: Comprehensive Benchmark Suite for Video Generative Models. web. https://vchitect.github.io/VBench-project/ (n.d.)
[9] VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness. web. https://vchitect.github.io/VBench-2.0-project/ (n.d.)
[10] VBench++: Comprehensive and Versatile Benchmark Suite for Video Generative Models. arxiv. https://arxiv.org/abs/2411.13503 (n.d.)
[11] AA-Video-T2V v2.0 Leaderboard - Top AI Video Models. web. https://artificialanalysis.ai/video/leaderboard/text-to-video (n.d.)
[12] MMMG. web. https://github.com/yaojh18/MMMG (2025-05-15T19:54:49.000Z)
[13] Simulating the Real World: A Unified Survey of Multimodal Generative Models. web. https://arxiv.org/html/2503.04641 (n.d.)
[14] Generative AI for multimodal content: a survey with empirical and experimental evaluations. web. https://link.springer.com/article/10.1007/s10462-026-11525-6 (2026-03-19)
[15] From Sora What We Can See: A Survey of Text-to-Video Generation. web. https://arxiv.org/html/2405.10674 (n.d.)
[16] A Survey of Generative Categories and Techniques in Multimodal Generative Models. web. https://arxiv.org/abs/2506.10016v3 (2025-05-29)
[17] Sora: A Review on Background, Technology, Limitations, and Opportunities of Large Vision Models. web. https://arxiv.org/html/2402.17177 (n.d.)
[18] Adaptive Caching for Faster Video Generation with Diffusion Transformers. hf-search. https://huggingface.co/papers/2411.02397 (2024-11-04)
[19] RainFusion: Adaptive Video Generation Acceleration via Multi-Dimensional Visual Redundancy. hf-search. https://huggingface.co/papers/2505.21036 (2025-05-27)
[20] LLMs Meet Multimodal Generation and Editing: A Survey. web. https://arxiv.org/html/2405.19334v1 (n.d.)
[21] UniVG: Towards UNIfied-modal Video Generation. hf-search. https://huggingface.co/papers/2401.09084 (2024-01-17)
[22] VBench Video Generation Leaderboard Benchmark Scores & AI Model Leaderboard | BenchmarkList. web. https://benchmarklist.com/benchmarks/vbench_video_generation/ (n.d.)
[23] ChronoMagic-Bench: A Benchmark for Metamorphic Evaluation of Text-to-Time-lapse Video Generation. hf-search. https://huggingface.co/papers/2406.18522 (2024-06-26)
[24] UEval: A Benchmark for Unified Multimodal Generation. hf-search. https://huggingface.co/papers/2601.22155 (2026-01-29)
[25] Sora as a World Model? A Complete Survey on Text-to-Video Generation. web. https://arxiv.org/html/2403.05131v3 (n.d.)
[26] Foundations & Trends in Multimodal Machine Learning: Principles, Challenges, and Open Questions. web. https://dl.acm.org/doi/full/10.1145/3656580 (2024-06-22)
