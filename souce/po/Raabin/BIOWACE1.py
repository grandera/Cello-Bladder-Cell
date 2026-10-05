import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb
import open_clip
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (classification_report, confusion_matrix,
                             balanced_accuracy_score, accuracy_score,
                             roc_auc_score)
from PIL import Image
import random

# ── Config ─────────────────────────────────────────────────────────────────
PREFIX          = "BIOWACE1"
MODEL_NAME      = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
DATA_DIR        = "/raid/DATASETS/po/Raabin"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE      = 32
CLASS_NAMES     = ["Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]
NUM_CLASSES     = len(CLASS_NAMES)
PROMPT_TEMPLATE = "a microscopy image of {} white blood cell"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# ── Dataset ────────────────────────────────────────────────────────────────
class RaabinDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)

def load_raabin_split(split_dir):
    paths, labels = [], []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(split_dir, cls_name)
        if not os.path.exists(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                paths.append(os.path.join(cls_dir, fname))
                labels.append(cls_idx)
    return paths, np.array(labels)

# ── Load model ─────────────────────────────────────────────────────────────
clip_model, _, preprocess_val = open_clip.create_model_and_transforms(MODEL_NAME)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)
clip_model.to(DEVICE)
clip_model.eval()

prompts     = [PROMPT_TEMPLATE.format(c.lower()) for c in CLASS_NAMES]
text_tokens = tokenizer(prompts, context_length=256).to(DEVICE)
with torch.no_grad():
    text_emb = clip_model.encode_text(text_tokens)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

# ── Load data ──────────────────────────────────────────────────────────────
TEST_DIR = os.path.join(DATA_DIR, "Test-A")
test_paths, test_labels_idx = load_raabin_split(TEST_DIR)
print(f"Test samples: {len(test_paths)}")
for i, c in enumerate(CLASS_NAMES):
    print(f"  {c}: {(test_labels_idx==i).sum()}")

test_loader = DataLoader(
    RaabinDataset(test_paths, test_labels_idx, transform=preprocess_val),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ── Evaluation ─────────────────────────────────────────────────────────────
wandb.init(
    project="BIOWA",
    name=f"{PREFIX}_zero_shot",
    config={
        "model": MODEL_NAME, "labels": CLASS_NAMES,
        "prompt_template": PROMPT_TEMPLATE, "step": f"{PREFIX}_zero_shot",
    }
)

logit_scale = clip_model.logit_scale
test_probs_list, test_true_list = [], []
with torch.no_grad():
    for batch_images, batch_labels in test_loader:
        batch_images = batch_images.to(DEVICE)
        img_features = clip_model.encode_image(batch_images)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        logits       = logit_scale.exp() * (img_features @ text_emb.t())
        test_probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
        test_true_list.append(batch_labels.numpy())

y_score = np.vstack(test_probs_list)
y_true  = np.concatenate(test_true_list)
y_pred  = np.argmax(y_score, axis=1)

final_bal_acc   = balanced_accuracy_score(y_true, y_pred)
final_wacc      = accuracy_score(y_true, y_pred)
final_macro_auc = roc_auc_score(np.eye(NUM_CLASSES)[y_true], y_score, average="macro")
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)

print(f"\nZero-shot Raabin-WBC Results:")
print(f"  BACC:      {final_bal_acc:.4f}")
print(f"  Accuracy:  {final_wacc:.4f}")
print(f"  Macro AUC: {final_macro_auc:.4f}")
print(f"  Macro F1:  {report['macro avg']['f1-score']:.4f}")

with open(f"{PREFIX}_report.json", "w") as f:
    json.dump(report, f, indent=2)

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm / cm.sum(axis=1, keepdims=True) * 100, cmap='Blues', vmin=0, vmax=100)
plt.colorbar(im, ax=ax)
ax.set_xticks(np.arange(NUM_CLASSES)); ax.set_yticks(np.arange(NUM_CLASSES))
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
ax.set_yticklabels(CLASS_NAMES)
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        ax.text(j, i, f"{cm[i,j]/cm[i].sum()*100:.1f}%", ha='center', va='center',
                color='white' if cm[i,j]/cm[i].sum() > 0.5 else 'black')
ax.set_title(f"{PREFIX} Zero-Shot — Confusion Matrix (%)")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.tight_layout()
cm_fname = f"{PREFIX}_cm.png"
plt.savefig(cm_fname, dpi=100, bbox_inches='tight')
plt.close()

per_class_f1   = {f"f1_{c}":        report[c]["f1-score"]  for c in CLASS_NAMES}
per_class_prec = {f"precision_{c}": report[c]["precision"] for c in CLASS_NAMES}
per_class_rec  = {f"recall_{c}":    report[c]["recall"]    for c in CLASS_NAMES}
per_class_auc  = {}
for i, cls in enumerate(CLASS_NAMES):
    if len(np.unique(y_true == i)) > 1:
        per_class_auc[f"auc_{cls}"] = roc_auc_score((y_true==i).astype(int), y_score[:,i])

wandb.log({
    "final_balanced_accuracy": final_bal_acc,
    "final_accuracy":          final_wacc,
    "final_macro_auc":         final_macro_auc,
    "final_macro_f1":          report['macro avg']['f1-score'],
    "confusion_matrix":        wandb.Image(cm_fname),
    **per_class_f1, **per_class_prec, **per_class_rec, **per_class_auc,
})
wandb.finish()
print("Done.")
