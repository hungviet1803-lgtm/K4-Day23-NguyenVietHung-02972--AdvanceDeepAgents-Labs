# World Models: Foundations, Architectures, Evaluation, and Recent Shifts
## TL;DR
- World models are learned internal models of environment dynamics that support prediction, imagination, and planning; the canonical early idea is to train an agent in its own learned “dream” and transfer the policy back to reality [1][2].
- The field’s core design choice is not whether to model the world, but what to model: latent dynamics for control and planning, explicit video/pixel prediction for perception-heavy settings, or abstract task-relevant predictions for search-based planning [3][4][5][6].
- Evaluation is fragmented but recurring metrics are clear: control papers report success rate, reward, ATE/RPE, and navigation error, while long-horizon prediction papers report FID/FVD/LPIPS/PSNR or multiple-choice accuracy in world-model benchmarks [7][8][9][10].
- Since 2024, world models have shifted toward foundation-style, multimodal, and action-controllable systems such as Sora, Genie 2, and Cosmos, but the community still disputes whether large video generators are true world models or only partial simulators [11][12][13][14].
- Recent work also emphasizes time-awareness, stronger memory, and reinforcement learning over the world-model objective itself, suggesting that long-horizon consistency and controllability remain the main bottlenecks [8][15][9][16].

## Background
A world model is a learned internal model of how an environment evolves over time. In reinforcement learning, this usually means learning latent dynamics, reward prediction, and sometimes observation reconstruction so that an agent can plan by imagination or search rather than by expensive interaction alone [1][17]. The classic motivating idea is sample-efficient control: if the agent can simulate futures inside the model, it can train policies with fewer real-environment episodes and even learn inside a “hallucinated dream” before transferring behavior back to the actual task [1][2][4].

The foundational line of work established two enduring patterns. First, a model can be used as a compact latent simulator, as in PlaNet and Dreamer, where action-conditioned rollouts occur in latent space rather than pixels [3][18][4][6]. Second, a model can support planning rather than full prediction, as in MuZero, which learns only task-relevant quantities—value, reward, and policy—sufficient for tree search [5]. Surveys of model-based RL later formalized this split into model learning and planning-learning integration, and emphasized uncertainty handling, differentiability, and the choice of how the learned model will be used [17].

## Latent Dynamics and Imagination-Based Control
The most established world-model family is latent dynamics modeling. PlaNet learns a dynamics model from images and chooses actions through fast online planning in latent space; its key selling point is that the agent no longer needs a known environment model, only learned latent transition structure [3][18]. Dreamer extends this idea by learning a recurrent state-space model with representation, transition, observation, and reward components, then optimizing an actor-critic by backpropagating through imagined trajectories in latent space [4][6][19]. In this paradigm, the world model is not merely predictive; it is a differentiable simulator for learning behavior.

This family tends to favor compactness and scalability over perfect observation reconstruction. Dreamer’s design separates imagination from decision-making by using actor and value networks over imagined latent rollouts, while PlaNet uses online optimization at decision time [18][4][6][20]. The architectural trade-off is simple: more explicit planning can improve transparency but costs inference-time compute, whereas amortized actor-critic policies are faster but rely on the latent model being accurate enough for long-horizon credit assignment [4][6][20].

## Abstract Prediction and Search-Oriented Models
A second family uses world models more abstractly. MuZero learns an internal model that predicts only quantities relevant to planning—policy, value, and reward—rather than reconstructing the full environment [5]. This is important because it decouples “world modeling” from pixel fidelity: the model can be imperfect in observation space yet still strong for decision-making, as long as the abstract rollout supports search [5][20]. The result was superhuman play on Go, chess, shogi, and strong Atari performance, demonstrating that a world model can be task-relevant rather than fully generative [5].

Recent surveys and comparison pages describe this as a distinct architectural choice: sequence and planning-oriented systems often model transition dynamics at a level sufficient for action selection, while omitting unnecessary details [21][22][20]. That distinction matters because it explains why some modern systems look less like simulators and more like learned planning kernels, even though they are still called world models [21][22].

## Evaluation Benchmarks and What They Measure
World-model evaluation is not standardized, but several recurring protocols dominate. For control, papers frequently use success rate, reward, ATE/RPE, navigation error, and rollout quality across a fixed suite of tasks. Time-Aware World Model evaluates nine Meta-World tasks—Assembly, Basketball, Box Close, Faucet Open, Hammer, Handle Pull, Lever Pull, Pick Out of Hole, and Sweep Into—and also three control-gym PDE tasks (Burgers, Allen–Cahn, Wave), using success rate and total episode reward [8][23]. In those experiments, TAWM reaches 100% success on Basketball at Δt = 2.5 ms and about 90% on Faucet Open at Δt = 50 ms, while baselines collapse at low observation rates [8][23].

For navigation and long-horizon visual prediction, MWM reports DreamSim, FID, ATE, RPE, success rate, and navigation error, including a 20.4% DreamSim reduction, 17.5% FID reduction, at least a 4× inference speedup, 10.9% ATE improvement, and 8.5% RPE improvement [15]. HippoCampus and HippoBench similarly emphasize long-rollout metrics—ATE RMSE, FID, FVD, PSNR, LPIPS—and show that hierarchical memory improves over a sliding-window baseline, with the best reported configuration reaching FVD 124.2, LPIPS 0.175, PSNR 24.96, and ATE 0.70 in one setting [9]. These papers make a common point: the hard part is not one-step prediction but stable multi-step consistency under long horizons [15][9].

For higher-level reasoning benchmarks, WorldPrediction uses a multiple-choice setup with visual observations, action equivalents, and counterfactual distractors, reporting best scores of 57.0% on WorldPrediction-WM and 38.1% on WorldPrediction-PP, while humans are perfect [7][24]. AutumnBench and WorldTest push further toward reward-free interaction followed by scored tests on 43 environments and 129 tasks, separating masked-frame prediction, planning, and causal-change prediction rather than conflating them into a single next-frame metric [10].

## Since 2024: Foundation-Style, Multimodal, and Action-Controllable World Models
Since 2024, the center of gravity has moved from “can video models predict the future?” to “can large foundation models simulate and control worlds?” OpenAI’s Sora description explicitly frames scalable video generation as a path toward general-purpose physical simulators, while also acknowledging failures in physics and long-duration coherence [11]. DeepMind’s Genie 2 takes the next step by describing a large-scale foundation world model that generates playable 3D environments from a prompt image and can simulate action consequences [12]. NVIDIA’s Cosmos platform similarly targets physical AI, packaging diffusion and autoregressive models trained on 9,000 trillion tokens, with versions between 4 and 14 billion parameters [13][25].

This newer line of work is broader than classical RL world modeling. It emphasizes multimodal pretraining, action-conditioned generation, interactive environments, and integration with robotics or autonomous driving [12][13][25]. At the same time, researchers still argue about whether pixel-level video generators are genuine world models or only partial simulators, especially when they lack explicit internal state or interpretable causal structure [11][14]. A recent survey frames the field as split between systems that understand the present state of the world and systems that predict future states for decision-making, reflecting this unresolved boundary [21].

## Trends and Open Problems
Three trends stand out since 2024. First, world models are becoming larger and more foundation-like, with broader pretraining corpora, more modalities, and stronger ties to interactive generation and physical AI [11][12][13][16]. Second, the field is moving from static evaluation toward task-grounded benchmarks that test control, planning, and causal reasoning under interaction, rather than only next-frame prediction [7][10]. Third, there is increasing interest in adapting the world-model objective itself through reinforcement learning or weak supervision, as in RLVR-World and related multimodal bootstrapping work [26][16].

The main open problems are still structural. Long-horizon consistency remains fragile, particularly for video and embodied settings [9][11][14]. Physics adherence is still imperfect even in large models, and current systems may produce plausible-looking but causally wrong futures [11][12][14]. Evaluation is also unsettled: a model can score well on one benchmark yet fail on another because the field lacks a single agreed notion of what a world model should optimize [7][8][10]. Finally, there is a conceptual debate over whether a world model must be explicitly stateful and action-conditioned, or whether sufficiently powerful generative video models already qualify [21][11][14].

## References
[1] World Models. arxiv. https://arxiv.org/abs/1803.10122 (n.d.)
[2] Recurrent World Models Facilitate Policy Evolution. web. https://worldmodels.github.io/ (2018-03-27)
[3] Learning Latent Dynamics for Planning from Pixels. arxiv. https://arxiv.org/abs/1811.04551 (n.d.)
[4] Dream to Control: Learning Behaviors by Latent Imagination. web. https://arxiv.org/html/1912.01603v2 (2019-12-03)
[5] Mastering Atari, Go, chess and shogi by planning with a learned model. web. https://www.nature.com/articles/s41586-020-03051-4 (2020-12-23)
[6] Introducing Dreamer: Scalable Reinforcement Learning Using World Models. web. https://research.google/blog/introducing-dreamer-scalable-reinforcement-learning-using-world-models/ (2020-03-18)
[7] WorldPrediction: A Benchmark for High-Level World Modeling and Long-Horizon Procedural Planning. arxiv. https://arxiv.org/abs/2506.04363 (n.d.)
[8] Time-Aware World Model for Adaptive Prediction and Control. web. https://proceedings.mlr.press/v267/nhu25a.html (2025-10-06)
[9] HippoCampus | Hierarchical Memory for Long-Horizon Video World Models. web. http://pengchensheng.com/project/hippocampus/index.html (n.d.)
[10] Benchmarking World-Model Learning. web. https://arxiv.org/html/2510.19788v2 (n.d.)
[11] Video generation models as world simulators. web. https://openai.com/index/video-generation-models-as-world-simulators/ (2024-02-15)
[12] Genie 2: A large-scale foundation world model. web. https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/ (2024-12-04)
[13] Advancing Physical AI with NVIDIA Cosmos World Foundation Model Platform. web. https://developer.nvidia.com/blog/advancing-physical-ai-with-nvidia-cosmos-world-foundation-model-platform/ (2025-01-09)
[14] Video generation models as world simulators. web. https://artificialcognition.net/posts/video-generation-world-simulators/ (2024-03-01)
[15] MWM: Mobile World Models for Action-Conditioned Consistent Prediction. web. https://arxiv.org/html/2603.07799 (n.d.)
[16] Bootstrapping World Models from Dynamics Models in Multimodal Foundation Models. hf-search. https://huggingface.co/papers/2506.06006 (2025-06-06)
[17] Model-based Reinforcement Learning: A Survey. arxiv. https://arxiv.org/abs/2006.16712 (2022-03-31)
[18] Learning Latent Dynamics for Planning from Pixels. web. https://planetrl.github.io/ (2019-02-15)
[19] Mastering diverse control tasks through world models. web. https://www.nature.com/articles/s41586-025-08744-2 (2025-04-02)
[20] DreamerV3 vs MuZero | Comparison | world-models.io. web. https://world-models.io/en/compare/dreamer-v3-vs-muzero/ (n.d.)
[21] Understanding World or Predicting Future? A Comprehensive Survey of World Models. web. https://dl.acm.org/doi/10.1145/3746449 (2025-09-09)
[22] World Models: A Comprehensive Survey of .... arxiv. https://arxiv.org/abs/2606.00133 (n.d.)
[23] Time-Aware World Model for Adaptive Prediction and Control. web. https://raw.githubusercontent.com/mlresearch/v267/main/assets/nhu25a/nhu25a.pdf (n.d.)
[24] WorldPrediction: A Benchmark for High-level World Modeling and Long-horizon Procedural Planning. web. https://worldprediction.github.io/ (n.d.)
[25] Cosmos World Foundation Models Openly Available to Physical AI Developers. web. https://blogs.nvidia.com/blog/cosmos-world-foundation-models/ (2025-01-06)
[26] RLVR-World: Training World Models with Reinforcement Learning. hf-search. https://huggingface.co/papers/2505.13934 (2025-05-20)
