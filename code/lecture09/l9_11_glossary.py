"""l9_11_glossary.py -- the bilingual glossary for Lecture 9.

Every line carries an English comment. The script writes the machine-readable term table
(Chinese, English, symbol, formula, note, part) that feeds the two glossary frames, the
site table and the printable Word/Excel handout.
"""
import pandas as pd                                                # build and persist the table
from _common_l9 import dump, OUT                                   # shared helpers

# Columns: term_zh, term_en, symbol, formula, note, part.
TERMS = [                                                               # term list
    ("深度学习", "deep learning", "", "", "多层可微网络的表示学习", "1 版图"),                        # concept
    ("表示学习", "representation learning", "", "", "自动从原始输入学特征", "1 版图"),                # concept
    ("端到端", "end-to-end", "", "", "从原始输入直接优化到输出", "1 版图"),                            # concept
    ("计算机视觉", "computer vision", "", "", "让机器理解图像与视频", "1 版图"),                        # concept
    ("图像分类", "image classification", "", "", "给整张图一个类别", "1 版图"),                        # concept
    ("目标检测", "object detection", "", "", "定位并分类多个物体", "1 版图"),                          # concept
    ("语义分割", "semantic segmentation", "", "", "给每个像素一个类别", "1 版图"),                     # concept
    ("数据驱动", "data-driven approach", "", "", "用数据学函数，而非手写规则", "2 分类"),               # knn
    ("最近邻", "nearest neighbour", "", "", "取距离最近的训练样本的标签", "2 分类"),                    # knn
    ("k 近邻", "k-nearest neighbours (k-NN)", "", "", "取最近 k 个投票或平均", "2 分类"),              # knn
    ("距离度量", "distance metric", "", "", "L1/L2/余弦；决定什么算相近", "2 分类"),                     # knn
    ("维度灾难", "curse of dimensionality", "", r"$\mathrm{nn}/\mathrm{fn}\to1$", "高维下距离失去区分度", "2 分类"),   # knn
    ("超参数", "hyper-parameter", "", "", "训练前设定，例如 k、学习率", "2 分类"),                     # knn
    ("验证集", "validation set", "", "", "选超参数用，不能是测试集", "2 分类"),                         # knn
    ("参数化方法", "parametric approach", "", "", "用参数概括数据，不存全部样本", "2 分类"),             # linear
    ("线性分类器", "linear classifier", "", r"$s=Wx+b$", "用线性打分做分类", "2 分类"),                # linear
    ("得分函数", "score function", "", r"$s=Wx$", "把输入映射到各类得分", "2 分类"),                   # linear
    ("softmax", "softmax", "", r"$e^{z_k}/\sum_j e^{z_j}$", "把得分变成概率", "2 分类"),              # linear
    ("交叉熵", "cross-entropy", "", r"$-\sum_k y_k\log p_k$", "分类的负对数似然", "2 分类"),           # linear
    ("温度", "temperature", r"$T$", "", "softmax 的锐度旋钮", "2 分类"),                              # linear
    ("激活函数", "activation function", "", r"$h=\phi(Wx+b)$", "引入非线性", "3 网络"),               # network
    ("sigmoid", "sigmoid", "", r"$1/(1+e^{-z})$", "压到 (0,1)，易饱和", "3 网络"),                    # network
    ("ReLU", "rectified linear unit (ReLU)", "", r"$\max(0,z)$", "不饱和、计算便宜", "3 网络"),        # network
    ("GELU", "Gaussian error linear unit (GELU)", "", "", "Transformer 常用激活", "3 网络"),          # network
    ("反向传播", "backpropagation", "", r"$\partial L/\partial w$", "用链式法则从输出往输入求梯度", "3 网络"),   # network
    ("链式法则", "chain rule", "", "", "复合函数的求导规则", "3 网络"),                                # network
    ("计算图", "computational graph", "", "", "把运算记成图，便于自动求导", "3 网络"),                  # network
    ("自动微分", "automatic differentiation", "", "", "由计算图自动算梯度（autograd）", "3 网络"),        # network
    ("梯度消失", "vanishing gradient", "", "", "深层网络梯度趋近 0", "3 网络"),                        # network
    ("梯度爆炸", "exploding gradient", "", "", "梯度爆炸导致发散", "3 网络"),                          # network
    ("学习率", "learning rate", r"$\eta$", "", "步长；过大发散、过小慢", "3 网络"),                    # network
    ("动量", "momentum", "", r"$v_t=\beta v_{t-1}+g_t$", "累积惯性抑制震荡", "3 网络"),               # network
    ("Adam", "Adam", "", "", "自适应步长 + 偏差校正", "3 网络"),                                     # network
    ("权重衰减", "weight decay", r"$\lambda$", "", "L2 正则的等价写法", "3 网络"),                    # network
    ("dropout", "dropout", "", "", "训练时随机丢弃神经元", "3 网络"),                                # network
    ("早停", "early stopping", "", "", "验证损失不再下降就停", "3 网络"),                             # network
    ("批归一化", "batch normalisation", "", "", "按批标准化激活，稳定训练", "3 网络"),                 # network
    ("初始化", "initialisation", "", "", "Xavier/He 控制初始方差", "3 网络"),                         # network
    ("卷积", "convolution", "", r"$(f*g)(t)$", "局部加权求和，参数共享", "4 CNN"),                    # conv
    ("卷积核", "kernel / filter", "", "", "可学习的小模板", "4 CNN"),                                # conv
    ("步长", "stride", "", "", "滑动步幅", "4 CNN"),                                                 # conv
    ("填充", "padding", "", "", "边界补零以保持尺寸", "4 CNN"),                                       # conv
    ("通道", "channel", "", "", "特征图的深度", "4 CNN"),                                            # conv
    ("感受野", "receptive field", "", "", "一个输出能看到的输入范围", "4 CNN"),                        # conv
    ("池化", "pooling", "", "", "下采样，扩大感受野", "4 CNN"),                                       # conv
    ("残差连接", "residual connection", "", r"$y=x+F(x)$", "恒等捷径，缓解深层退化", "4 CNN"),        # conv
    ("迁移学习", "transfer learning", "", "", "用预训练权重做新任务", "4 CNN"),                        # conv
    ("数据增强", "data augmentation", "", "", "翻转/裁剪/扰动，免费的正则", "4 CNN"),                  # conv
    ("生成对抗网络", "generative adversarial network (GAN)", "", "", "生成器与判别器对抗", "4 CNN"),    # conv
    ("扩散模型", "diffusion model", "", "", "学去噪过程来生成样本", "4 CNN"),                          # conv
    ("循环神经网络", "recurrent neural network (RNN)", "", "", "按时间步递归处理序列", "5 序列"),      # seq
    ("长短期记忆", "long short-term memory (LSTM)", "", "", "门控结构，缓解梯度消失", "5 序列"),        # seq
    ("序列到序列", "sequence to sequence (seq2seq)", "", "", "编码器-解码器结构", "5 序列"),          # seq
    ("注意力", "attention", "", r"$\alpha=\mathrm{softmax}(q^\top k)$", "按相关性加权求和", "5 序列"),  # seq
    ("自注意力", "self-attention", "", r"$\mathrm{softmax}(QK^\top/\sqrt{d})V$", "序列内部互相看", "5 序列"),   # seq
    ("多头注意力", "multi-head attention", "", "", "多组投影并行，学不同关系", "5 序列"),               # seq
    ("位置编码", "positional encoding", "", "", "给序列注入顺序信息", "5 序列"),                       # seq
    ("Transformer", "Transformer", "", "", "纯注意力架构，可并行", "5 序列"),                         # seq
    ("编码器-解码器", "encoder-decoder", "", "", "编码输入、解码输出", "5 序列"),                       # seq
    ("掩码语言模型", "masked language model", "", "", "预测被遮住的词（BERT）", "5 序列"),              # seq
    ("自回归", "autoregressive", "", "", "按前面预测下一个（GPT）", "5 序列"),                         # seq
    ("预训练", "pretraining", "", "", "在大规模无标注数据上训练表示", "6 微调"),                       # finetune
    ("微调", "fine-tuning", "", "", "在预训练模型上继续训练", "6 微调"),                              # finetune
    ("线性探测", "linear probing", "", "", "冻结主干、只训练分类头", "6 微调"),                        # finetune
    ("提示学习", "prompting", "", "", "用提示词调用模型，不改参数", "6 微调"),                         # finetune
    ("LoRA", "low-rank adaptation (LoRA)", "", r"$W+BA$", "低秩适配器，只训练少数参数", "6 微调"),     # finetune
    ("QLoRA", "quantised LoRA (QLoRA)", "", "", "主干量化到 4-bit 再挂 LoRA", "6 微调"),               # finetune
    ("混合精度", "mixed precision", "", "", "fp16/bf16 与 fp32 混用", "6 微调"),                      # finetune
    ("梯度累积", "gradient accumulation", "", "", "多个小批累积后再更新", "6 微调"),                   # finetune
    ("量化", "quantisation", "", "", "用低比特表示权重/激活", "6 微调"),                              # finetune
    ("知识蒸馏", "knowledge distillation", "", "", "用大模型指导小模型", "6 微调"),                    # finetune
    ("图神经网络", "graph neural network (GNN)", "", "", "在图上做消息传递", "6 微调"),                # finetune
    ("强化学习", "reinforcement learning (RL)", "", "", "通过交互与回报学习策略", "6 微调"),           # finetune
    ("对齐", "alignment", "", "", "让模型行为符合人类意图与规范", "6 微调"),                          # finetune
    ("红队测试", "red teaming", "", "", "主动攻击以发现模型缺陷", "6 微调"),                          # finetune
    ("灾难性遗忘", "catastrophic forgetting", "", "", "微调后丢失原有能力", "6 微调"),                 # finetune
    ("上下文长度", "context length", "", "", "模型一次能处理的 token 数", "6 微调"),                    # finetune
]                                                                       # end of the term list

# ---------------------------------------------------------------- persist the table
glossary = pd.DataFrame(TERMS, columns=["term_zh", "term_en", "symbol", "formula", "note", "part"])   # build the frame
dump(glossary, "l9_glossary.csv")                                  # machine-readable CSV in the data folder
counts = glossary["part"].value_counts().sort_index()              # terms per part
print("glossary terms:", len(glossary))                            # report the total
print(counts.to_string())                                          # report the split by part
assert len(glossary) >= 60, "the glossary should hold at least sixty terms"   # acceptance check
md_lines = ["# Lecture 9 glossary", "",                            # markdown title
            "| 术语 (zh) | Term (en) | Symbol | Formula | Note | Part |",   # header row
            "| --- | --- | --- | --- | --- | --- |"]                # separator row
for _, r in glossary.iterrows():                                   # one row per term
    md_lines.append("| %s | %s | %s | %s | %s | %s |" % (r["term_zh"], r["term_en"],   # Chinese and English
                                                         r["symbol"], r["formula"], r["note"], r["part"]))   # rest
(OUT / "l9_glossary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")   # markdown table
print("[artifact]", OUT / "l9_glossary.md")                        # log the markdown path
print("done: glossary with %d terms" % len(glossary))              # closing log
