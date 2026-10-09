# LLM Agents and Tool Use: Architectures, Evaluation, and Open Problems
## TL;DR
- LLM agents are best understood as control loops around a model: plan or reason, choose tools, execute actions, observe results, and update the next step; early systems differ mainly in how they trigger tool use and route work to external modules [1][2][3][4][5][6]
- Tool use improves access to current or specialized information, helps with computation and action, and can make behavior more interpretable, but it also introduces orchestration overhead and failure points [1][3][6][7]
- Training and orchestration now span prompt-only ReAct-style agents, self-supervised tool-trace learning, function-calling APIs, reflective repair loops, and self-play RL for tool calling [3][8][9][10][11]
- Evaluation has moved from single success rates toward outcome metrics plus trajectory-aware, reliability, safety, and compute-normalized measures such as pass^k, win rate, tool-order accuracy, and production-like stress tests [12][13][14][15][16][17][18][19][20][21][22][23][24][11]
- The main unresolved questions are whether scaffolds help after compute normalization, how to benchmark long-horizon memory and planning, and how to keep agents safe against prompt injection, brittle APIs, and saturated benchmarks [19][20][21][22][23][7][24][11]

## Background
LLM agents are systems that use a language model not just to answer, but to act: they plan, invoke tools or external modules, observe outcomes, and revise their next step. Foundational frameworks frame this as augmenting the model with external experts or modules, so it can route tasks to calculators, retrievers, search engines, or domain services instead of relying only on parametric knowledge [1][3][4][6]. ReAct made the reasoning-acting loop explicit by interleaving thoughts and actions, while Toolformer showed that tool use can be learned self-supervised from a small number of demonstrations [2][3].

This matters because tool use addresses several limits of a pure LLM: stale knowledge, weak arithmetic or symbolic execution, and poor access to live or specialized systems [1][3][6]. It also changes the shape of evaluation and deployment, because success now depends on the model, the prompt or controller, the tool interface, and the environment state [5][6][13][14].

## Foundations and early agent patterns
A useful way to classify early agent systems is by how they connect the model to tools. MRKL routes inputs to specialist modules through a router, emphasizing modularity, fallback, and compositionality [1]. ReAct keeps the model “in the loop” by alternating reasoning traces with actions, which helps the model track and update plans while interacting with environments or knowledge bases [2]. HuggingGPT uses ChatGPT as a controller that plans, selects models, executes tasks, and then synthesizes a response from the results [4].

The shared design principle is a closed loop: generate a plan, act, observe, and revise. Surveys and practitioner guides now describe this loop with components such as an agent core, planning module, tools, and memory module [5][6]. More recent reviews also separate passive tool use such as retrieval from autonomous invocation and feedback learning, which is helpful because these modes differ substantially in how much control the LLM has over when and why tools are called [5][25][5].

## Architectures and training methods
Modern tool-use systems mostly differ in orchestration. One family uses prompt-only control and function-calling APIs: the model decides whether to call a tool, the runtime executes it, and the result is fed back for another turn [13][26]. Another family learns tool calling from traces. Toolformer trains the model to decide which APIs to call, when to call them, and what arguments to pass, using self-supervision and only a small amount of demonstration data [3].

A third family adds explicit reflection or hierarchical search over tools. AnyTool combines a hierarchical API retriever, solver, and self-reflection, and reports a +35.4% average pass-rate gain over ToolLLM on ToolBench [9]. Surveys of planning describe reflection and refinement as a way to repair failed plans, while noting that RL-based planners often need many environment interactions [8]. Recent work also pushes toward self-play RL for tool learning from zero data, suggesting that tool competence can be improved without a large labeled corpus [11].

The practical architecture trend is toward large tool inventories and structured selection. Function-calling APIs explicitly allow zero, one, or multiple tool calls, often with JSON-schema-defined functions and deferred loading of rare tools [13][26]. That helps scale to large tool sets, but it also shifts the burden to tool retrieval, invocation policies, and error handling [9][13][20].

## Benchmarking and evaluation
Early tool benchmarks mostly scored whether the final call or answer was correct. API-Bank, for example, evaluates call accuracy, response quality, and three task levels: Call, Retrieve+Call, and Plan+Retrieve+Call; it reports 59.40%/63.66% Call correctness for GPT-3.5-turbo/GPT-4 and only 22.00%/70.00% on the hardest setting [12]. ToolEval uses pass rate and win rate, while BFCL adds serial and parallel function calls, multiple languages, hallucination checks, and agentic evaluation with AST- and execution-based scoring [13][15].

Newer benchmarks are more trajectory-aware. TRAJECT-Bench reports trajectory exact match, inclusion, tool-usage, satisfy, and final accuracy, and shows a large gap between simple and hard queries for Claude-4: 0.846 EM / 0.905 Acc on simple queries versus 0.445 EM / 0.517 Acc on hard ones [16]. MTU-BENCH expands this further with multi-turn, multi-tool, and out-of-distribution settings, adding metrics like tool number accuracy and tool order accuracy [17]. These metrics matter because tool-use errors are often about the path, not just the endpoint [16][17].

Robustness and stability are now first-class evaluation concerns. τ-bench measures repeated-trial reliability with pass^k and reports that the best gpt-4o function-calling agent exceeds 60% average task success but falls below 25% at pass^8 in retail [14]. StableToolBench was introduced because real API status changes made ToolBench unstable over time; it replaces ad hoc evaluation with stable pass/win variants and API simulators [18]. In parallel, BFCL and related docs distinguish outcome scores from diagnostic trajectory signals, so that the leaderboard score reflects task success rather than every intermediate action [13][26].

## Trends and open problems
The biggest shift since 2024 is from “can the model use a tool?” to “does the whole scaffold work in realistic conditions?” Recent surveys say the field now emphasizes memory, reliability, safety, cost, and production-like stress conditions, rather than only end-to-end task success [19][24][11]. This also explains why benchmarks increasingly include dynamic environments, multi-turn interactions, and fault injection [14][18][22].

A major dispute is whether agent scaffolds help once compute is normalized. The compute-normalized view argues that planning, verification, memory, tool use, and role separation should be compared per model call, tool call, token, dollar, and failure regime, because raw success can hide ceiling effects or expensive over-orchestration [20][21][7]. That concern shows up in production anecdotes about malformed tool calls, timeouts, infinite loops, wrong plans, and cost spirals, especially for long trajectories and large context windows [21][7].

Safety is now inseparable from tool use. AgentDojo and InjecAgent show that prompt injection and other attacks are especially dangerous when agents execute tools over untrusted data, and that current defenses are incomplete [22][23]. At the same time, evaluation coverage remains uneven: coarse success rates can miss intermediate failures, policy violations, or unsafe tool selections [19][22].

The open research agenda is therefore fairly clear. We still need better memory benchmarks [24], more realistic reliability testing [11], cost-aware evaluation [19][20], and benchmark families that stay useful as tools and APIs evolve [19][7]. More broadly, the field needs a cleaner separation between the capability of the base model and the quality of the surrounding agent harness, because many reported gains may come from scaffolding rather than better underlying tool reasoning [20][21][24].

## References
[1] MRKL Systems: A modular, neuro-symbolic architecture that combines large language models, external knowledge sources and discrete reasoning. arxiv. https://arxiv.org/abs/2205.00445 (n.d.)
[2] ReAct: Synergizing Reasoning and Acting in Language Models. web. https://arxiv.org/abs/2210.03629 (2023-03-10)
[3] Toolformer: Language Models Can Teach Themselves to Use Tools. arxiv. https://arxiv.org/abs/2302.04761 (2023-02-09)
[4] HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face. web. https://arxiv.org/abs/2303.17580 (2023-03-30)
[5] A Review of Prominent Paradigms for LLM-Based Agents: Tool Use (Including RAG), Planning, and Feedback Learning. web. https://arxiv.org/html/2406.05804 (n.d.)
[6] Introduction to LLM Agents. web. https://developer.nvidia.com/blog/introduction-to-llm-agents/ (2023-11-30)
[7] Agentic Workflows Are Harder Than You Think - Jeff Adler. web. https://jads.app/blog/agentic-workflows-are-harder-than-you-think (2024-10-11T19:30:38.000Z)
[8] Understanding the planning of LLM agents: A survey. arxiv. https://arxiv.org/abs/2402.02716 (2024-02-05)
[9] AnyTool: Self-Reflective, Hierarchical Agents for Large-Scale API Calls. arxiv. https://arxiv.org/abs/2402.04253 (2024-02-06)
[10] FlowBench: Revisiting and Benchmarking Workflow-Guided Planning for LLM-based Agents. hf-search. https://huggingface.co/papers/2406.14884 (2024-06-21)
[11] ReliabilityBench: Evaluating LLM Agent Reliability Under Production-Like Stress Conditions. hf-search. https://huggingface.co/papers/2601.06112 (2026-01-03)
[12] API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs. arxiv. https://arxiv.org/abs/2304.08244 (n.d.)
[13] The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models. web. https://raw.githubusercontent.com/mlresearch/v267/main/assets/patil25a/patil25a.pdf (n.d.)
[14] $τ$-bench: A Benchmark for Tool-Agent-User Interaction with Safety and Reliability Metrics. arxiv. https://arxiv.org/abs/2406.12045 (n.d.)
[15] ToolEval Leaderboard. web. https://openbmb.github.io/ToolBench/ (n.d.)
[16] TRAJECT-Bench:A Trajectory-Aware Benchmark for Evaluating Agentic Tool Use. web. https://exa.ai/library/publication/1rwb8gwkl59 (2025-10-06)
[17] MTU-BENCH: A Multi-Granularity Tool-Use Benchmark for Large Language Models. web. https://proceedings.iclr.cc/paper_files/paper/2025/file/4d13b2d99519c5415661dad44ab7edcd-Paper-Conference.pdf (n.d.)
[18] StableToolBench: Towards Stable Large-Scale Benchmarking on Tool Learning of Large Language Models. web. https://aclanthology.org/anthology-files/pdf/findings/2024.findings-acl.664.pdf (n.d.)
[19] A Survey on Evaluation of LLM-based Agents. web. https://aclanthology.org/2026.findings-acl.1330.pdf (n.d.)
[20] When Do Agent Scaffolds Actually Help? Toward Compute-Normalized Evaluation. web. https://srijan-open.github.io/agent-workflow-scaffolds-preprint/paper.pdf (n.d.)
[21] Benchmarking Single Agent Performance. web. https://www.langchain.com/blog/react-agent-benchmarking (2025-02-10T17:13:25.000Z)
[22] AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents. web. https://proceedings.neurips.cc/paper_files/paper/2024/file/97091a5177d8dc64b1da8bf3e1f6fb54-Paper-Datasets_and_Benchmarks_Track.pdf (n.d.)
[23] InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents. web. http://arxiv.org/abs/2403.02691 (2024-08-04)
[24] Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions. hf-search. https://huggingface.co/papers/2507.05257 (2025-07-07)
[25] From Language to Action: A Review of Large Language Models as Autonomous Agents and Tool Users. hf-search. https://huggingface.co/papers/2508.17281 (2025-10-28)
[26] docs/evaluation.md. web. https://github.com/sierra-research/tau2-bench/blob/HEAD/docs/evaluation.md (n.d.)
