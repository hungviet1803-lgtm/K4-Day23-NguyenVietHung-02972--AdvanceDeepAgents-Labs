# Reinforcement Learning for LLM Reasoning

## TL;DR
- RL for LLM reasoning is best understood as post-training that uses reward signals to improve multi-step problem solving, often after or alongside SFT; in 2024–2025, verifiable-reward RL and GRPO-style training became the dominant recipe for reasoning models. [1][2][3]
- DPO-style methods are important contrast cases: they optimize preferences without explicit reward models or an RL loop, so they are usually treated as RL-free preference optimization rather than reasoning RL proper. [4][5][6][7]
- The main technical shift has been away from critic-heavy PPO toward simpler grouped or sample-relative updates such as GRPO and RLOO, paired with either sparse final-answer rewards or denser process supervision. [8][9][10][11]
- Evaluation is still fragile: math and reasoning benchmarks often show high seed variance, while newer benchmarks such as RewardMATH and broader competition-style suites try to make robustness and contamination harder to ignore. [12][13][14][15]
- Since 2024, the field has moved toward verifier-guided RL, off-policy guidance, and hybrid train-time/test-time scaling, but reliable feedback for hard tasks remains the central bottleneck. [16][17][18][19]

## Background
Reinforcement learning for LLM reasoning refers to post-training methods that improve a language model using rewards tied to correctness, usefulness, or reasoning quality, rather than only imitating demonstrations. In the reasoning setting, the goal is not just fluent answers but better multi-step problem solving, self-correction, and robustness under difficult questions. Recent systems frame this as an RL stage after SFT, or in some cases as pure RL with verifiable rewards when the final answer can be checked automatically. [1][2][3]

This matters because reasoning tasks expose limits of supervised fine-tuning: a model can learn to imitate traces without learning how to search, verify, or recover from errors. RL is attractive when the environment supplies a reward signal that is closer to the task objective, such as final-answer correctness on math problems or rule-based checks for code and logic. At the same time, preference-optimization methods like DPO and IPO are often positioned as alternatives that avoid the instability and cost of a full RL loop, which is why they are useful comparators when defining the RL-for-reasoning space. [4][5][6][7]

## Core training recipes and reward design
A central pattern in recent reasoning models is the move from PPO-style RLHF to lighter-weight policy-gradient variants. DeepSeekMath describes GRPO as a PPO variant that removes the critic and uses the average reward over multiple sampled completions as the baseline; in that work, RL improved GSM8K from 82.9% to 88.2% and MATH from 46.8% to 51.7%. Back to Basics makes a similar case for REINFORCE and RLOO, arguing that PPO is unnecessarily complicated for a pre-trained LLM environment and that leave-one-out baselines can work well without a learned value function. [8][9]

Reward design is the other major axis. The simplest setup is outcome supervision: reward only the final answer, especially when a verifier can check correctness automatically. DeepSeek-R1 explicitly uses rule-based rewards for final-answer correctness and format compliance, and it avoids neural outcome or process reward models partly because of reward-hacking concerns. In contrast, process supervision assigns credit step by step, which is appealing for reasoning because it gives denser feedback about where a solution goes wrong. [1][10][11][2]

The evidence base for process supervision is strong but still limited. Let’s Verify Step by Step and the OpenAI process-supervision work both report that step-level feedback outperforms final-answer-only supervision on math reasoning, and PRM800K established a large human-labeled step dataset for that line of work. Newer surveys summarize the same split as outcome reward models versus process reward models, with rule-based rewards gaining prominence in the reasoning-RL era. [10][11][13]

A practical consequence is that many modern pipelines combine sparse verifier signals with auxiliary structure: format rewards, rejection sampling, cold-start SFT, or model-induced process labels. AutoPSV and MiPS are examples of reducing annotation cost by generating process supervision from verifier confidence or model-induced traces, which points to a broader trend toward cheaper reward data rather than bigger reward models alone. [17][18]

## Benchmarks and empirical evaluation
Most math-oriented RL-for-reasoning papers still benchmark on AIME, AMC, MATH500, HMMT, and BRUMO, usually with pass@1 or avg@n metrics. ReTool evaluates on AIME2024 and AIME2025 and reports both answer accuracy and tool-use diagnostics such as code ratio, code lines, code pass rate, and invocation timing, showing that RL can improve not only final correctness but also the policy’s tool-usage pattern. [12][13][15]

AceReason-Nemotron extends this math-centric view by combining math and code benchmarks: AIME2024, AIME2025, MATH500, HMMT2025 Feb, BRUMO2025, LiveCodeBench v5/v6, EvalPlus, and Codeforces ELO or percentile from LiveCodeBench Pro. It reports that math-only RL improves AIME2025 by +14.6% for the 7B model and +17.2% for the 14B model, while also improving LiveCodeBench v5 by +6.8% and +5.8%. The message is that reasoning RL may transfer across math and code, but only when evaluation captures both domains and uses enough samples to stabilize estimates. [12]

That caveat matters because evaluation noise is large. A Sober Look at Progress in Language Model Reasoning re-evaluates AIME’24, AMC’23, and MATH500 with repeated seeds and statistical tests, finding large seed-driven swings in pass@1 and concluding that many reported RL gains are not statistically significant on distillation-based models. It also warns that models tuned on older benchmarks can lose ground on newer ones like AIME’25, which suggests overfitting to the benchmark distribution rather than genuine reasoning improvements. [13]

Newer robustness-oriented benchmarks try to address these weaknesses directly. RewardMATH evaluates reward-model quality with one correct and nine incorrect solutions per problem, using classification accuracy and MRR, and reports that its rankings correlate better with optimized policy performance than older RewardBench-style setups. MathArena broadens coverage to competition-style math and proof tasks, including exact-match scoring for final-answer problems, human grading for proof-based tasks, and code execution for Project Euler, while ARC-AGI emphasizes generalization from few examples rather than standard math accuracy. [14][20][15][21]

## Trends and open problems
Since 2024, the most visible trend has been the rise of reasoning-specific post-training systems that explicitly combine train-time RL with test-time compute. OpenAI’s o1 post emphasizes that performance improves with both more reinforcement learning and more thinking at inference time, and DeepSeek-R1 similarly presents reasoning as something that can be incentivized through pure RL with GRPO and verifiable rewards. These systems helped normalize the idea that reasoning quality is partly a scaling problem, not just an architecture problem. [3][1][16]

A second trend is the expansion of verifier-guided and process-supervised methods. AutoPSV and MiPS try to automate step-level labels from verifier behavior, while the foundational PRM work showed why stepwise signals matter in the first place. This direction is attractive because it reduces reliance on scarce human annotations, but it still assumes that the verifier is trustworthy and that process labels correlate with actual reasoning quality. [10][11][17][18]

A third trend is the search for cheaper and more flexible optimization than classic on-policy RLHF. GPT-NeoX’s 2024 post highlights support for DPO, KTO, reward-model training, and future REINFORCE/online RL, showing how open stacks are broadening the post-training toolkit. At the same time, off-policy guidance such as LUFFY suggests that pure on-policy RLVR may leave useful reasoning traces on the table; the reported gains over six math benchmarks and on out-of-distribution tasks indicate that mixed-policy training could be a practical next step. [22][19]

The open problems are familiar but sharper now. First, reliable feedback is still hard for complex tasks where automatic verification is incomplete, and DeepSeek-R1 explicitly notes that such tasks make pure RL difficult to scale. Second, reward hacking remains a concern whenever the reward model is imperfect or the signal is too sparse. Third, evaluation is still not standardized enough: high-variance math benchmarks can exaggerate small differences, so future work needs better protocols, stronger contamination checks, and more tasks that measure transfer beyond narrow contest math. [1][2][13][14][15]

## References
[1] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arxiv. https://arxiv.org/abs/2501.12948 (2025-01-20)
[2] Beyond Supervised Fine Tuning: How Reinforcement Learning Empowers AI with Minimal Labels. web. https://fireworks.ai/blog/reinforcement-learning-with-verifiable-reward (2025-01-27)
[3] Learning to reason with LLMs. web. https://openai.com/index/learning-to-reason-with-llms/ (2024-09-12)
[4] A Survey of Direct Preference Optimization. arxiv. https://arxiv.org/abs/2503.11701 (n.d.)
[5] A Comprehensive Survey of Direct Preference Optimization. arxiv. https://arxiv.org/abs/2410.15595 (n.d.)
[6] Direct Preference Optimization with an Offset. arxiv. https://arxiv.org/abs/2402.10571 (n.d.)
[7] IPO: Your Language Model is Secretly a Preference Classifier. arxiv. https://arxiv.org/abs/2502.16182 (n.d.)
[8] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arxiv. https://arxiv.org/abs/2402.03300 (2024-04-27)
[9] Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs. arxiv. https://arxiv.org/abs/2402.14740 (2024-02-23)
[10] Let's Verify Step by Step. arxiv. https://arxiv.org/abs/2305.20050 (2023-05-31)
[11] Sailing by the Stars: A Survey on Reward Models and Learning Strategies for Learning from Rewards. arxiv. https://arxiv.org/abs/2505.02686 (2025-06-12)
[12] ReTool: Reinforcement Learning for Strategic Tool Use in LLMs. arxiv. https://arxiv.org/abs/2504.11536 (n.d.)
[13] A Sober Look at Progress in Language Model Reasoning - arXiv. arxiv. https://arxiv.org/abs/2504.07086 (n.d.)
[14] Evaluating Robustness of Reward Models for Mathematical Reasoning. hf-search. https://huggingface.co/papers/2410.01729 (2024-10-02)
[15] MathArena: Evaluating LLMs on Uncontaminated Math Competitions. web. https://www.proceedings.com/content/085/085713-0679open.pdf (n.d.)
[16] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. web. https://arxiv.org/html/2501.12948 (2026-01-04)
[17] AutoPSV: Automated Process-Supervised Verifier. web. https://proceedings.neurips.cc/paper_files/paper/2024/file/9246aa822579d9b29a140ecdac36ad60-Paper-Conference.pdf (n.d.)
[18] Multi-step Problem Solving Through a Verifier: An Empirical Analysis on Model-induced Process Supervision. web. https://aclanthology.org/2024.findings-emnlp.429.pdf (n.d.)
[19] Learning to reason under Off-Policy Guidance. web. https://proceedings.neurips.cc/paper_files/paper/2025/file/a9d5c33e38442450f9781bd7bad3c81e-Paper-Conference.pdf (n.d.)
[20] Evaluating Robustness of Reward Models for Mathematical Reasoning. web. https://rewardmath-rewardmath-project.static.hf.space/index.html (n.d.)
[21] ARC-AGI-1. web. https://arcprize.org/arc-agi/1 (n.d.)
[22] RLHF and RLAIF in GPT-NeoX. web. https://blog.eleuther.ai/rlhf-and-rlaif-in-gpt-neox/ (2024-10-10)
