"""l10_11_glossary.py -- the bilingual glossary for Lecture 10.

Every line carries an English comment. The script writes the machine-readable term table
(Chinese, English, symbol, formula, note, part) that feeds the two glossary frames, the site
table and the printable Word/Excel handout.
"""
import pandas as pd                                                # build and persist the table
from _common_l10 import dump, OUT                                  # shared helpers

# Columns: term_zh, term_en, symbol, formula, note, part.
TERMS = [                                                               # term list
    ("文本分析", "text analytics", "", "", "从非结构化文本中提取结构化信息", "1 版图"),              # concept
    ("文本作为数据", "text as data", "", "", "把语言当作可测量的变量", "1 版图"),                    # concept
    ("语料库", "corpus", "", "", "按用途组织的大量文本", "1 版图"),                                  # concept
    ("标注", "annotation", "", "", "人工给出的标签或结构", "1 版图"),                                # concept
    ("标注一致性", "inter annotator agreement", "", "", "多人标注之间的吻合程度", "1 版图"),          # concept
    ("分词", "word segmentation", "", "", "把连续文本切成词（中文必需）", "1 版图"),                 # concept
    ("归一化", "normalisation", "", "", "统一大小写、全半角、繁简等", "1 版图"),                      # concept
    ("停用词", "stopword", "", "", "高频但信息量低的词", "1 版图"),                                  # concept
    ("词干提取", "stemming", "", "", "粗暴截断词尾（英文）", "1 版图"),                              # concept
    ("词形还原", "lemmatisation", "", "", "还原到词典原形", "1 版图"),                              # concept
    ("未登录词", "out-of-vocabulary (OOV)", "", "", "词表里没有的词", "1 版图"),                     # concept
    ("隐私", "privacy", "", "", "文本中的个人信息与脱敏", "1 版图"),                                 # concept
    ("音系学", "phonetics and phonology", "", "", "语言的声音层面", "2 语言"),                       # ling
    ("形态学", "morphology", "", "", "词的内部结构", "2 语言"),                                      # ling
    ("词性标注", "part-of-speech tagging", "", "", "给每个词标词类", "2 语言"),                      # ling
    ("句法", "syntax", "", "", "词如何组成句子", "2 语言"),                                          # ling
    ("依存句法", "dependency parsing", "", "", "词与词之间的依存关系", "2 语言"),                     # ling
    ("语义学", "semantics", "", "", "词与句子的意义", "2 语言"),                                      # ling
    ("语篇", "discourse", "", "", "跨句子的结构与指代", "2 语言"),                                    # ling
    ("歧义", "ambiguity", "", "", "同一形式多种解释", "2 语言"),                                      # ling
    ("稀疏", "sparsity", "", "", "语言组合无限，数据永远覆盖不全", "2 语言"),                         # ling
    ("变体", "variation", "", "", "同义表达、方言、错别字", "2 语言"),                                # ling
    ("符号主义", "symbolic approach", "", "", "规则与语法驱动", "2 语言"),                            # ling
    ("统计方法", "statistical approach", "", "", "语料 + 概率模型", "2 语言"),                        # ling
    ("连接主义", "connectionist approach", "", "", "神经网络 + 表示学习", "2 语言"),                  # ling
    ("n 元语法", "n-gram", "", "", "连续 n 个词作为一个单位", "3 表示"),                              # repr
    ("词袋", "bag of words", "", "", "只统计词频，忽略顺序", "3 表示"),                               # repr
    ("TF-IDF", "term frequency-inverse document frequency", "", r"$\mathrm{tf}\cdot\log\frac{N}{\mathrm{df}}$", "词频乘以逆文档频率", "3 表示"),   # repr
    ("文档-词矩阵", "document-term matrix", "", "", "行为文档、列为词的稀疏矩阵", "3 表示"),            # repr
    ("哈希技巧", "hashing trick", "", "", "用哈希函数省去词表", "3 表示"),                             # repr
    ("one-hot", "one-hot encoding", "", "", "每个词一个正交向量", "3 表示"),                          # repr
    ("分布假设", "distributional hypothesis", "", "", "上下文相似则词义相似", "3 表示"),               # repr
    ("词向量", "word vector / embedding", "", "", "把词映射到稠密向量", "3 表示"),                     # repr
    ("Word2vec", "word2vec", "", "", "用上下文预测训练词向量", "4 词向量"),                           # w2v
    ("跳字模型", "skip-gram", "", "", "用中心词预测上下文", "4 词向量"),                              # w2v
    ("连续词袋", "continuous bag of words (CBOW)", "", "", "用上下文预测中心词", "4 词向量"),          # w2v
    ("负采样", "negative sampling", "", r"$U^{3/4}$", "用少数负例替代全词表 softmax", "4 词向量"),    # w2v
    ("共现矩阵", "co-occurrence matrix", "", "", "窗口内词对的计数矩阵", "4 词向量"),                 # w2v
    ("奇异值分解", "singular value decomposition (SVD)", "", r"$X=U\Sigma V^\top$", "降维得到稠密向量", "4 词向量"),   # w2v
    ("GloVe", "global vectors (GloVe)", "", "", "用共现概率的比值编码语义", "4 词向量"),               # w2v
    ("余弦相似度", "cosine similarity", "", r"$\frac{a\cdot b}{\lVert a\rVert\lVert b\rVert}$", "向量夹角的余弦", "4 词向量"),   # w2v
    ("类比", "analogy", "", r"$a-b+c$", "用向量运算检验语义关系", "4 词向量"),                        # w2v
    ("内在评估", "intrinsic evaluation", "", "", "只评表示本身（相似度/类比）", "4 词向量"),            # w2v
    ("外在评估", "extrinsic evaluation", "", "", "把表示放进下游任务评效果", "4 词向量"),              # w2v
    ("一词多义", "polysemy", "", "", "同一个词多个义项", "4 词向量"),                                 # w2v
    ("上下文嵌入", "contextual embedding", "", "", "同一个词在不同语境下向量不同", "4 词向量"),        # w2v
    ("文本分类", "text classification", "", "", "给整段文本一个标签", "5 任务"),                       # task
    ("情感分析", "sentiment analysis", "", "", "判断正负中性", "5 任务"),                             # task
    ("主题模型", "topic model", "", "", "从文档集合中提主题", "5 任务"),                              # task
    ("潜在狄利克雷分配", "latent Dirichlet allocation (LDA)", "", "", "生成式主题模型", "5 任务"),      # task
    ("命名实体识别", "named entity recognition (NER)", "", "", "识别人名/机构/金额/日期", "5 任务"),    # task
    ("BIO 标注", "BIO tagging", "", "", "B-开始 / I-内部 / O-外部", "5 任务"),                        # task
    ("序列标注", "sequence labelling", "", "", "给序列每个位置打标签", "5 任务"),                      # task
    ("摘要", "summarisation", "", "", "抽取式或生成式压缩文本", "5 任务"),                             # task
    ("机器翻译", "machine translation", "", "", "跨语言转换", "5 任务"),                              # task
    ("问答", "question answering", "", "", "根据问题找答案", "5 任务"),                               # task
    ("检索增强生成", "retrieval-augmented generation (RAG)", "", "", "先检索再生成", "6 实践"),        # llm
    ("引用溯源", "citation tracing", "", "", "答案标注来源片段", "6 实践"),                            # llm
    ("提示", "prompt", "", "", "给模型的指令与上下文", "6 实践"),                                     # llm
    ("少样本", "few-shot", "", "", "在提示里给几个示例", "6 实践"),                                   # llm
    ("指令微调", "instruction tuning", "", "", "用指令-回答数据微调", "6 实践"),                      # llm
    ("提示版本管理", "prompt versioning", "", "", "把提示当代码管理", "6 实践"),                       # llm
    ("Cohen's kappa", "Cohen's kappa", "", "", "扣除随机一致的标注一致性", "6 实践"),                # llm
    ("领域词典", "domain lexicon", "", "", "自定义词表，改善分词与抽取", "6 实践"),                    # llm
    ("BIO 序列", "BIO sequence", "", "", "标签序列的表示", "6 实践"),                                 # llm
    ("泄漏", "leakage", "", "", "训练与测试共享信息", "5 任务"),                                      # task
    ("交叉验证", "cross-validation", "", "", "轮换留出验证集", "5 任务"),                             # task
    ("F1 分数", "F1 score", "", r"$2PR/(P+R)$", "精确率与召回率的调和平均", "5 任务"),                # task
    ("BLEU", "BLEU", "", "", "翻译的 n-gram 重合度", "5 任务"),                                       # task
    ("ROUGE", "ROUGE", "", "", "摘要的召回导向重合度", "5 任务"),                                     # task
    ("信息抽取", "information extraction", "", "", "从文本中抽出结构化字段", "5 任务"),               # task
    ("多原型", "multiple prototypes", "", "", "一个词学多个向量以表示多义", "4 词向量"),              # w2v
    ("窗口", "context window", "", "", "共现统计或预测时看的邻域大小", "4 词向量"),                    # w2v
    ("降维", "dimensionality reduction", "", "", "把高维表示压到低维", "3 表示"),                     # repr
    ("嵌入矩阵", "embedding matrix", "", "", "词表 × 维度的参数矩阵", "4 词向量"),                    # w2v
    ("预训练", "pretraining", "", "", "在大规模语料上先训练表示", "6 实践"),                          # llm
    ("微调", "fine-tuning", "", "", "在预训练模型上适配下游", "6 实践"),                              # llm
    ("中文分词歧义", "segmentation ambiguity", "", "", "不同切法改变语义", "1 版图"),                 # concept
    ("全半角", "full-width and half-width", "", "", "字符宽度差异需归一化", "1 版图"),                # concept
    ("繁简转换", "traditional-simplified conversion", "", "", "统一字形", "1 版图"),                  # concept
    ("脱敏", "de-identification", "", "", "去除可识别个人的信息", "6 实践"),                          # llm
]                                                                       # end of the term list

# ---------------------------------------------------------------- persist the table
glossary = pd.DataFrame(TERMS, columns=["term_zh", "term_en", "symbol", "formula", "note", "part"])   # build the frame
dump(glossary, "l10_glossary.csv")                                 # machine-readable CSV in the data folder
counts = glossary["part"].value_counts().sort_index()              # terms per part
print("glossary terms:", len(glossary))                            # report the total
print(counts.to_string())                                          # report the split by part
assert len(glossary) >= 60, "the glossary should hold at least sixty terms"   # acceptance check
md_lines = ["# Lecture 10 glossary", "",                           # markdown title
            "| 术语 (zh) | Term (en) | Symbol | Formula | Note | Part |",   # header row
            "| --- | --- | --- | --- | --- | --- |"]                # separator row
for _, r in glossary.iterrows():                                   # one row per term
    md_lines.append("| %s | %s | %s | %s | %s | %s |" % (r["term_zh"], r["term_en"],   # Chinese and English
                                                         r["symbol"], r["formula"], r["note"], r["part"]))   # rest
(OUT / "l10_glossary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")   # markdown table
print("[artifact]", OUT / "l10_glossary.md")                        # log the markdown path
print("done: glossary with %d terms" % len(glossary))              # closing log
