import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
from collections import Counter
import warnings

warnings.filterwarnings("ignore")

RUN_MODE = "baseline"      # "baseline" 或 "improved"

MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 8
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
GRAD_CLIP = 1.0
RANDOM_SEED = 42

VAL_SIZE = 0.15
TEST_SIZE = 0.15

LABEL_MAP = {"yes": 0, "no": 1, "maybe": 2}
LABEL_NAMES = ["yes", "no", "maybe"]

USE_CLASS_WEIGHT = (RUN_MODE == "improved")
SELECT_BY = "macro_f1" if RUN_MODE == "improved" else "accuracy"

CKPT_PATH = f"pubmedbert_{RUN_MODE}.pt"
RESULT_PATH = f"results_{RUN_MODE}.txt"

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch : {torch.__version__}")
print(f"Device  : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
print(f"模式    : {RUN_MODE}  (类别权重={USE_CLASS_WEIGHT}, 选型指标={SELECT_BY})")


print("\n加载 PubMedQA (pqa_labeled)...")
try:
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
except Exception:
    dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")

print(f"样本数  : {len(dataset)}")
print(f"标签分布: {dict(Counter(s['final_decision'] for s in dataset))}")


def build_text(sample):
    question = sample["question"].strip()
    passages = " ".join(sample["context"]["contexts"])
    return f"{question} [SEP] {passages}"


texts = [build_text(s) for s in dataset]
labels = [LABEL_MAP[s["final_decision"].lower()] for s in dataset]
labels_arr = np.array(labels)


trainval_idx, test_idx = train_test_split(
    np.arange(len(texts)),
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=labels_arr,
)
val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)
train_idx, val_idx = train_test_split(
    trainval_idx,
    test_size=val_ratio,
    random_state=RANDOM_SEED,
    stratify=labels_arr[trainval_idx],
)
print(f"训练集 : {len(train_idx)}  |  验证集 : {len(val_idx)}  |  测试集 : {len(test_idx)}")


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class PubMedQADataset(Dataset):
    def __init__(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, pos):
        idx = self.indices[pos]
        enc = tokenizer(
            texts[idx],
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels[idx], dtype=torch.long),
        }


train_loader = DataLoader(PubMedQADataset(train_idx), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(PubMedQADataset(val_idx), batch_size=BATCH_SIZE * 2, shuffle=False)
test_loader = DataLoader(PubMedQADataset(test_idx), batch_size=BATCH_SIZE * 2, shuffle=False)


train_counts = Counter([labels[i] for i in train_idx])
print(f"训练集类别分布: { {LABEL_NAMES[k]: v for k, v in sorted(train_counts.items())} }")

if USE_CLASS_WEIGHT:
    weights = torch.tensor(
        [len(train_idx) / (len(LABEL_NAMES) * train_counts[i]) for i in range(len(LABEL_NAMES))],
        dtype=torch.float,
    ).to(DEVICE)
    print(f"类别权重: { {LABEL_NAMES[i]: round(weights[i].item(), 3) for i in range(3)} }")
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
else:
    loss_fn = torch.nn.CrossEntropyLoss()


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_NAMES),
    id2label={i: l for i, l in enumerate(LABEL_NAMES)},
    label2id=LABEL_MAP,
).to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, int(total_steps * WARMUP_RATIO), total_steps
)

use_amp = DEVICE.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


def evaluate(loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(input_ids=ids, attention_mask=mask)
            preds += out.logits.argmax(-1).cpu().tolist()
            trues += batch["labels"].tolist()
    return np.array(trues), np.array(preds)


best_score, best_state, best_epoch = -1.0, None, -1
history = []

print(f"\n开始微调，共 {EPOCHS} 个 epoch（测试集全程不参与）\n")
for epoch in range(1, EPOCHS + 1):
    model.train()
    ep_loss = 0.0

    for batch in train_loader:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labs = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(input_ids=ids, attention_mask=mask)
            loss = loss_fn(out.logits, labs)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        ep_loss += loss.item()

    y_true, y_pred = evaluate(val_loader)
    v_acc = accuracy_score(y_true, y_pred)
    v_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)

    print(f"Epoch {epoch}/{EPOCHS}  loss={ep_loss/len(train_loader):.4f}  "
          f"val_acc={v_acc:.4f}  val_macro_f1={v_f1:.4f}  "
          f"[yes={per_class[0]:.3f} no={per_class[1]:.3f} maybe={per_class[2]:.3f}]")

    history.append({
        "epoch": epoch, "val_acc": v_acc, "val_macro_f1": v_f1,
        "f1_yes": per_class[0], "f1_no": per_class[1], "f1_maybe": per_class[2],
    })

    score = v_f1 if SELECT_BY == "macro_f1" else v_acc
    if score > best_score:
        best_score, best_epoch = score, epoch
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  -> 新最优 (val {SELECT_BY}={best_score:.4f})")

model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
torch.save(best_state, CKPT_PATH)
print(f"\n按验证集选中 epoch {best_epoch}，权重已保存: {os.path.abspath(CKPT_PATH)}")


y_true, y_pred = evaluate(test_loader)

acc = accuracy_score(y_true, y_pred)
mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
wf1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
report = classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=4)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

val_best = history[best_epoch - 1]

lines = [
    f"Run mode     : {RUN_MODE}",
    f"Class weight : {USE_CLASS_WEIGHT}   Model selection: val {SELECT_BY}",
    f"Split        : train {len(train_idx)} / val {len(val_idx)} / test {len(test_idx)}",
    f"Epochs       : {EPOCHS}   Batch size: {BATCH_SIZE}   LR: {LR}",
    f"Selected     : epoch {best_epoch}  (val acc={val_best['val_acc']:.4f}, "
    f"val macro_f1={val_best['val_macro_f1']:.4f})",
    "",
    "---- TEST SET (held out, evaluated once) ----",
    f"Accuracy    : {acc:.4f}",
    f"Macro F1    : {mf1:.4f}",
    f"Weighted F1 : {wf1:.4f}",
    "",
    report,
    "Confusion matrix (rows=true, cols=pred):",
    "          " + "  ".join(f"{l:>7}" for l in LABEL_NAMES),
]
for i, row in enumerate(cm):
    lines.append(f"  {LABEL_NAMES[i]:>5}  " + "  ".join(f"{v:7d}" for v in row))

lines.append("")
lines.append("---- Validation, per epoch ----")
for h in history:
    lines.append(
        f"  epoch {h['epoch']}: acc={h['val_acc']:.4f}  macro_f1={h['val_macro_f1']:.4f}  "
        f"f1_yes={h['f1_yes']:.3f}  f1_no={h['f1_no']:.3f}  f1_maybe={h['f1_maybe']:.3f}"
    )

text = "\n".join(lines)
print("\n" + text)

with open(RESULT_PATH, "w", encoding="utf-8") as f:
    f.write(text)
print(f"\n结果已保存: {os.path.abspath(RESULT_PATH)}")