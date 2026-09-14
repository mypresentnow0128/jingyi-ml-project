
import ast
import operator
import numpy as np

class LiteratureSearch:

    LOW_RELEVANCE = 0.30

    def __init__(self):
        self.ready = False
        self.corpus = []
        self.questions = []
        self.vectorizer = None
        self.matrix = None

    def sample_questions(self, n=5, seed=0):

        if not self.ready:
            self.load()
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.questions), size=min(n, len(self.questions)), replace=False)
        return [self.questions[i] for i in idx]

    def load(self):
        from datasets import load_dataset
        from sklearn.feature_extraction.text import TfidfVectorizer

        print("[LiteratureSearch] 加载语料...")
        try:
            ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        except Exception:
            ds = load_dataset("pubmed_qa", "pqa_labeled", split="train")

        for s in ds:
            self.questions.append(s["question"])
            self.corpus.append(" ".join(s["context"]["contexts"]))

        index_text = [f"{q} {c}" for q, c in zip(self.questions, self.corpus)]

        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
        self.matrix = self.vectorizer.fit_transform(index_text)
        self.ready = True
        print(f"[LiteratureSearch] 就绪，{len(self.corpus)} 篇摘要")

    def __call__(self, query: str, top_k: int = 3) -> str:
        if not self.ready:
            self.load()

        from sklearn.metrics.pairwise import cosine_similarity

        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix)[0]
        top_idx = np.argsort(scores)[::-1][:top_k]

        if len(top_idx) == 0 or scores[top_idx[0]] < 0.01:
            return "没有检索到相关文献。语料中可能不包含该主题，不要重复检索。"

        out = []
        for rank, i in enumerate(top_idx, 1):
            snippet = self.corpus[i][:600]
            out.append(f"[{rank}] (相关度 {scores[i]:.3f}) {snippet}...")

        body = "\n\n".join(out)

        if scores[top_idx[0]] < self.LOW_RELEVANCE:
            body += (
                f"\n\n[检索质量提示] 最高相关度仅 {scores[top_idx[0]]:.3f}，"
                f"语料库中很可能没有该主题的文献。不要再换关键词重试，"
                f"请直接基于以上内容调用 classify_answer，或说明证据不足。"
            )
        return body


class PubMedBERTClassifier:


    MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    LABELS = ["yes", "no", "maybe"]

    MIN_CONFIDENCE = 0.50
    MIN_MARGIN = 0.12

    def __init__(self, ckpt_path="pubmedbert_improved.pt"):
        self.ckpt_path = ckpt_path
        self.model = None
        self.tokenizer = None

    def load(self):
        import os
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(
                f"找不到权重 {self.ckpt_path}，先把 train_pubmedbert.py 跑完"
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_NAME, num_labels=3
        )
        state = torch.load(self.ckpt_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        print(f"[PubMedBERT] 权重已加载 ({self.device})")

    def __call__(self, question: str, context: str = "") -> str:
        import torch

        if self.model is None:
            self.load()

        text = f"{question.strip()} [SEP] {context.strip()}"
        enc = self.tokenizer(
            text, max_length=512, padding="max_length",
            truncation=True, return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()

        pred = self.LABELS[int(np.argmax(probs))]
        detail = ", ".join(f"{l}={p:.3f}" for l, p in zip(self.LABELS, probs))

        top2 = sorted(probs, reverse=True)[:2]
        max_p, margin = top2[0], top2[0] - top2[1]

        if max_p < self.MIN_CONFIDENCE or margin < self.MIN_MARGIN:
            return (
                f"[置信度不足] 各类概率: {detail}（最高 {max_p:.3f}，前两名差距 {margin:.3f}）。"
                f"模型无法区分，此结果不可作为结论使用。"
                f"请补充更相关的证据后重试，或在 Final Answer 中明确说明证据不足、无法判断。"
            )
        return f"预测: {pred} | 各类概率: {detail}"


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("不支持的表达式")


def calculator(expression: str) -> str:
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        return str(_safe_eval(tree.body))
    except Exception as e:
        return f"计算失败: {e}"


_search = LiteratureSearch()
_classifier = PubMedBERTClassifier()

TOOLS = {
    "search_literature": {
        "func": lambda q: _search(q),
        "desc": "在 PubMed 文献库中检索相关摘要。输入：英文检索关键词。返回：最相关的 3 段摘要。",
    },
    "classify_answer": {
        "func": lambda arg: _classifier(*_split_two(arg)),
        "desc": "用微调过的 PubMedBERT 判断医学问题的结论。输入格式：问题 || 证据文本。返回：yes/no/maybe 及各类概率。",
    },
    "calculator": {
        "func": calculator,
        "desc": "四则运算。输入：数学表达式，如 (45-30)/30*100",
    },
}


def _split_two(arg: str):

    if "||" in arg:
        q, c = arg.split("||", 1)
        return q.strip(), c.strip()
    return arg.strip(), ""


def render_tool_descriptions() -> str:

    return "\n".join(f"- {name}: {t['desc']}" for name, t in TOOLS.items())


def sample_real_questions(n=5, seed=0):

    return _search.sample_questions(n=n, seed=seed)


if __name__ == "__main__":
    print("=== 测试 calculator ===")
    print(calculator("(45-30)/30*100"))

    print("\n=== 测试 search_literature ===")
    print(_search("aerobic exercise cognitive function older adults")[:500])

    print("\n=== 测试 classify_answer（需要权重跑完）===")
    try:
        print(_classifier(
            "Does aerobic exercise improve cognitive function in older adults?",
            "Several randomized trials found significant improvements in executive function.",
        ))
    except FileNotFoundError as e:
        print(f"跳过: {e}")
