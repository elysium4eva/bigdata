# Lecture 11 glossary

| 术语 (zh) | Term (en) | Symbol | Formula | Note | Part |
| --- | --- | --- | --- | --- | --- |
| 大语言模型 | large language model (LLM) |  |  | 用海量文本训练的下一个词预测器 | 1 原理 |
| token | token |  |  | 模型处理的最小单位 | 1 原理 |
| 分词 | tokenisation |  |  | 把文本切成 token 的过程 | 1 原理 |
| BPE | byte-pair encoding |  |  | 常用子词切分算法 | 1 原理 |
| 词向量 | embedding | $e(w)\in\mathbb{R}^d$ |  | 把词映射成向量 | 1 原理 |
| Transformer | transformer |  |  | 靠注意力堆叠的序列模型 | 1 原理 |
| 注意力 | attention | $\mathrm{softmax}(QK^\top/\sqrt{d})V$ |  | 按相关性加权取信息 | 1 原理 |
| logits | logits |  |  | softmax 之前的原始分数 | 1 原理 |
| softmax | softmax |  | $p_i=e^{z_i}/\sum_j e^{z_j}$ | 把分数变成概率 | 1 原理 |
| 温度 | temperature | $T$ | $z_i/T$ | 越高越随机 | 1 原理 |
| top-p 采样 | nucleus sampling |  |  | 只在累积概率 p 内采样 | 1 原理 |
| 上下文窗口 | context window |  |  | 模型一次能看到的 token 数 | 1 原理 |
| 预训练 | pretraining |  |  | 在海量文本上学下一个词 | 1 原理 |
| 缩放定律 | scaling law |  | $L\propto N^{-\alpha}$ | loss 随规模幂律下降 | 1 原理 |
| 零样本 | zero-shot |  |  | 不给例子直接做 | 2 提示 |
| 少样本 | few-shot |  |  | 给几个例子再做 | 2 提示 |
| 提示工程 | prompt engineering |  |  | 把任务写清楚并给出格式 | 2 提示 |
| 系统提示 | system prompt |  |  | 设定角色与规则的前置指令 | 2 提示 |
| 结构化输出 | structured output |  |  | 要求 JSON/表格等固定格式 | 2 提示 |
| 思维链 | chain-of-thought (CoT) |  |  | 先写步骤再给结论 | 2 提示 |
| 自洽投票 | self-consistency |  |  | 多次采样取多数 | 2 提示 |
| 反思 | reflexion |  |  | 让模型检查并修改自己的输出 | 2 提示 |
| 提示注入 | prompt injection |  |  | 文档里的指令劫持模型 | 2 提示 |
| 角色扮演 | role prompting |  |  | 让模型扮演某个身份作答 | 2 提示 |
| 监督微调 | supervised fine-tuning (SFT) |  |  | 用示范答案教它照做 | 3 对齐 |
| 人类反馈强化学习 | RLHF |  |  | 奖励模型 + 策略优化 | 3 对齐 |
| 奖励模型 | reward model |  |  | 学人类偏好打分的模型 | 3 对齐 |
| PPO | proximal policy optimisation |  |  | 常用的策略优化算法 | 3 对齐 |
| 直接偏好优化 | DPO |  |  | 跳过奖励模型直接优化偏好 | 3 对齐 |
| 可验证奖励 | RLVR |  |  | 用答案对错当奖励 | 3 对齐 |
| 推理模型 | reasoning model |  |  | 先想再答的模型 | 3 对齐 |
| 测试时计算 | test-time compute |  |  | 推理时多花算力 | 3 对齐 |
| 蒸馏 | distillation |  |  | 用大模型教小模型 | 3 对齐 |
| 对齐税 | alignment tax |  |  | 对齐带来的能力损失 | 3 对齐 |
| 数据污染 | data contamination |  |  | 测试集混进了训练集 | 3 对齐 |
| 函数调用 | function calling |  |  | 模型输出要调用什么工具 | 4 工程 |
| 工具调用 | tool use |  |  | 让模型使用外部能力 | 4 工程 |
| 智能体 | agent |  |  | 模型 + 工具 + 循环 | 4 工程 |
| 规划 | planning |  |  | 把任务拆成步骤 | 4 工程 |
| 记忆 | memory |  |  | 跨轮次保存的上下文 | 4 工程 |
| MCP | model context protocol |  |  | 工具接入的标准协议 | 4 工程 |
| 检索增强生成 | retrieval-augmented generation (RAG) |  |  | 先检索再生成 | 4 工程 |
| 切分 | chunking |  |  | 把长文档切成小块 | 4 工程 |
| 向量数据库 | vector database |  |  | 存向量并做相似检索 | 4 工程 |
| 重排 | reranking |  |  | 对召回结果再排序 | 4 工程 |
| 混合检索 | hybrid retrieval |  |  | 关键词 + 向量一起用 | 4 工程 |
| BM25 | BM25 |  |  | 经典关键词打分 | 4 工程 |
| 幻觉 | hallucination |  |  | 编造看似合理但不真实的内容 | 5 风险 |
| 护栏 | guardrail |  |  | 生成之后的校验与拦截 | 5 风险 |
| 评测集 | evaluation set |  |  | 用来打分的固定题目 | 5 风险 |
| 准确率 | accuracy |  |  | 答对的比率 | 5 风险 |
| 拒答率 | refusal rate |  |  | 不知道时说不知道的比率 | 5 风险 |
| 格式合规率 | format compliance |  |  | 输出符合 schema 的比率 | 5 风险 |
| LLM 裁判 | LLM-as-judge |  |  | 用模型给模型打分 | 5 风险 |
| 内容标识 | content provenance |  |  | 标注 AI 生成内容 | 5 风险 |
| 深度伪造 | deepfake |  |  | 合成的音视频 | 5 风险 |
| 版权 | copyright |  |  | 训练与生成内容的权利问题 | 5 风险 |
| 隐私 | privacy |  |  | 输入里的个人信息 | 5 风险 |
| 成本 | token cost |  |  | 按 token 计价的费用 | 6 落地 |
| 延迟 | latency |  |  | 从请求到首字的时间 | 6 落地 |
| 并发 | throughput |  |  | 单位时间能处理的请求数 | 6 落地 |
| 流式输出 | streaming |  |  | 边生成边返回 | 6 落地 |
| 本地部署 | on-premise deployment |  |  | 把模型放在自己机房 | 6 落地 |
| 微调 | fine-tuning |  |  | 在自己的数据上继续训练 | 6 落地 |
| LoRA | low-rank adaptation |  |  | 低成本微调方法 | 6 落地 |
| 量化 | quantisation |  |  | 用更低位宽压缩模型 | 6 落地 |
| 蒸馏部署 | distilled deployment |  |  | 用小模型上线 | 6 落地 |
| A/B 测试 | A/B test |  |  | 用实验比较两个方案 | 6 落地 |
| 增量测量 | incremental measurement |  |  | 只看 AI 带来的净增量 | 6 落地 |
| 人机协同 | human-in-the-loop |  |  | 关键环节保留人工 | 6 落地 |
| 模型卡 | model card |  |  | 记录模型能力与限制 | 6 落地 |
| 提示模板 | prompt template |  |  | 可复用的提示骨架 | 6 落地 |
| 上下文工程 | context engineering |  |  | 管理塞进窗口的内容 | 6 落地 |
| 评测驱动开发 | eval-driven development |  |  | 先写评测再改系统 | 6 落地 |
| 沙箱 | sandbox |  |  | 限制工具权限的执行环境 | 6 落地 |
| 白名单 | allowlist |  |  | 只允许约定的输入与工具 | 6 落地 |
| 可追溯 | traceability |  |  | 每一步都能回放 | 6 落地 |
| 失效开关 | kill switch |  |  | 出问题时能立刻停用 | 6 落地 |
| 基线 | baseline |  |  | 先做简单版本再比 | 6 落地 |
| 别用 LLM 做算术 | do not let an LLM do arithmetic |  |  | 交给工具 | 6 落地 |
| AI 是放大器 | AI is an amplifier |  |  | 放大你的判断，不替代它 | 6 落地 |
