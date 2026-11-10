import random
import os, json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, accuracy_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import wandb
import open_clip

from dataloader_vit import split_dataset, CellDataset, all_labels

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

PREFIX          = "BIOCE1"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR        = "/raid/DATASETS/pc/data"
CLASS_NAMES     = all_labels
NUM_CLASSES     = len(CLASS_NAMES)
MODEL_NAME      = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
BATCH_SIZE      = 32
PROMPT_TEMPLATE = "a microscopy image of {} bladder cancer cells"

clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(MODEL_NAME)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)
clip_model.to(DEVICE).eval()

prompts     = [PROMPT_TEMPLATE.format(c) for c in CLASS_NAMES]
text_tokens = tokenizer(prompts, context_length=256).to(DEVICE)
with torch.no_grad():
    text_emb = clip_model.encode_text(text_tokens)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

train_paths, train_labels, test_paths, test_labels = split_dataset(
    DATA_DIR, total_samples=None, train_ratio=0.8)
test_labels_idx = np.array([np.argmax(l) for l in test_labels])

test_loader = DataLoader(
    CellDataset(test_paths, test_labels_idx, transform=preprocess_val),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

wandb.init(
    project="BIOCE",
    name=f"{PREFIX}_zero_shot",
    config={
        "model": MODEL_NAME, "labels": CLASS_NAMES,
        "prompt_template": PROMPT_TEMPLATE, "step": f"{PREFIX}_zero_shot",
    }
)

all_scores, all_true = [], []
with torch.no_grad():
    for batch_images, batch_labels in test_loader:
        img_emb = clip_model.encode_image(batch_images.to(DEVICE))
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        probs   = torch.softmax(clip_model.logit_scale.exp() * (img_emb @ text_emb.t()), dim=1)
        all_scores.append(probs.cpu().numpy())
        all_true.append(batch_labels.numpy())

y_score = np.vstack(all_scores)
y_true  = np.concatenate(all_true)
y_pred  = np.argmax(y_score, axis=1)

cm = confusion_matrix(y_true, y_pred)
cm_percent = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True) * 100
plt.figure(figsize=(8, 6))
plt.imshow(cm_percent, interpolation="nearest", cmap=plt.cm.Blues)
plt.title(f"{PREFIX} Zero-Shot — Confusion Matrix (%)")
plt.colorbar()
plt.xticks(np.arange(NUM_CLASSES), CLASS_NAMES, rotation=45, ha="right")
plt.yticks(np.arange(NUM_CLASSES), CLASS_NAMES)
for i in range(cm_percent.shape[0]):
    for j in range(cm_percent.shape[1]):
        plt.text(j, i, f"{cm_percent[i,j]:.1f}%", ha="center", va="center",
                 color="white" if cm_percent[i,j] > 50 else "black", fontsize=8)
plt.tight_layout()
plt.savefig(f"{PREFIX}_cm.png", dpi=300)
plt.close()

auc_scores = {}
plt.figure(figsize=(10, 8))
for i, label in enumerate(CLASS_NAMES):
    if len(np.unique(y_true == i)) > 1:
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_score[:, i])
        auc = roc_auc_score((y_true == i).astype(int), y_score[:, i])
        auc_scores[label] = auc
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"{PREFIX} Zero-Shot ROC Curves")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{PREFIX}_roc.png", dpi=300)
plt.close()

final_macro_auc = roc_auc_score(np.eye(NUM_CLASSES)[y_true], y_score, average="macro")
bal_acc         = balanced_accuracy_score(y_true, y_pred)
weighted_acc    = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
with open(f"{PREFIX}_report.json", "w") as f:
    json.dump(report, f, indent=2)

per_class_auc  = {"auc_" + k: v for k, v in auc_scores.items()}
per_class_f1   = {"f1_" + c: report[c]["f1-score"] for c in CLASS_NAMES}
per_class_prec = {"precision_" + c: report[c]["precision"] for c in CLASS_NAMES}
per_class_rec  = {"recall_" + c: report[c]["recall"] for c in CLASS_NAMES}

wandb.log({
    "final_balanced_accuracy": bal_acc,
    "final_macro_auc": final_macro_auc,
    "final_weighted_accuracy": weighted_acc,
    "final_macro_f1": report["macro avg"]["f1-score"],
    "final_weighted_f1": report["weighted avg"]["f1-score"],
    "final_macro_precision": report["macro avg"]["precision"],
    "final_weighted_precision": report["weighted avg"]["precision"],
    "final_macro_recall": report["macro avg"]["recall"],
    "final_weighted_recall": report["weighted avg"]["recall"],
    **per_class_auc, **per_class_f1, **per_class_prec, **per_class_rec,
    "confusion_matrix": wandb.Image(f"{PREFIX}_cm.png"),
    "roc_curves": wandb.Image(f"{PREFIX}_roc.png"),
    "report": wandb.Table(
        columns=["class", "precision", "recall", "f1", "auc"],
        data=[[c, report[c]["precision"], report[c]["recall"], report[c]["f1-score"],
               auc_scores.get(c, 0.0)] for c in CLASS_NAMES]
    ),
})
wandb.finish()
print(f"\nTest Balanced Accuracy : {bal_acc:.4f}")
print(f"Test Macro AUC         : {final_macro_auc:.4f}")
