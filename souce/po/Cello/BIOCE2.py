import random
import os, json
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, accuracy_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import open_clip
from PIL import Image

from dataloader_vit import split_dataset, CellDataset, all_labels

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

PREFIX          = "BIOCE2"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR        = "/raid/DATASETS/pc/data"
CLASS_NAMES     = all_labels
NUM_CLASSES     = len(CLASS_NAMES)
MODEL_NAME      = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
BATCH_SIZE      = 32
NUM_EPOCHS      = 100
PATIENCE        = 10
LR_HEAD         = 1e-4
PROMPT_TEMPLATE = "a microscopy image of {} bladder cancer cells"

clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(MODEL_NAME)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)
clip_model.to(DEVICE)

prompts     = [PROMPT_TEMPLATE.format(c) for c in CLASS_NAMES]
text_tokens = tokenizer(prompts, context_length=256).to(DEVICE)
with torch.no_grad():
    text_emb = clip_model.encode_text(text_tokens)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)


class BiomedAttentionMap:
    """Attention map for BiomedCLIP timm-based ViT using Q/K hooks."""
    def __init__(self, device):
        self.device = device
        self.q = self.k = None
        self.hook_handles = []

    def _register_hooks(self, visual_trunk):
        last_attn = visual_trunk.blocks[-1].attn
        def q_hook(m, i, o): self.q = o.detach()
        def k_hook(m, i, o): self.k = o.detach()
        self.hook_handles.append(last_attn.q_norm.register_forward_hook(q_hook))
        self.hook_handles.append(last_attn.k_norm.register_forward_hook(k_hook))

    def _remove_hooks(self):
        for h in self.hook_handles: h.remove()
        self.hook_handles = []

    def generate(self, image_tensor, model_visual):
        image_tensor = image_tensor.to(self.device)
        trunk = model_visual.visual.trunk
        self._register_hooks(trunk)
        with torch.no_grad():
            x = trunk.patch_embed(image_tensor)
            x = trunk._pos_embed(x)
            x = trunk.norm_pre(x)
            for block in trunk.blocks:
                x = block(x)
        self._remove_hooks()
        if self.q is None or self.k is None:
            return np.zeros((224, 224))
        q = self.q
        k = self.k
        scale = q.shape[-1] ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        attn = attn.mean(dim=1)
        cls_attn = attn[0, 0, 1:].cpu()
        cls_attn = cls_attn - cls_attn.min()
        if cls_attn.max() > 0:
            cls_attn = cls_attn / cls_attn.max()
        gs     = int(cls_attn.shape[0] ** 0.5)
        cam    = cls_attn.reshape(gs, gs).numpy()
        cam_up = torch.nn.functional.interpolate(
            torch.tensor(cam).unsqueeze(0).unsqueeze(0),
            size=(224, 224), mode="bilinear", align_corners=False
        ).squeeze().numpy()
        return cam_up


def generate_attention_grid(model, test_paths, y_true, y_pred, class_names, device, prefix, n_per_class=2):
    attn_map   = BiomedAttentionMap(device)
    wandb_imgs = {}
    for cls_idx, cls_name in enumerate(class_names):
        correct_idx   = [i for i in range(len(y_true)) if y_true[i] == cls_idx and y_pred[i] == cls_idx]
        incorrect_idx = [i for i in range(len(y_true)) if y_true[i] == cls_idx and y_pred[i] != cls_idx]
        if len(correct_idx) == 0: continue
        np.random.seed(42)
        sampled_correct   = np.random.choice(correct_idx, min(n_per_class, len(correct_idx)), replace=False)
        sampled_incorrect = np.random.choice(incorrect_idx, min(n_per_class, len(incorrect_idx)), replace=False) if incorrect_idx else []
        all_sampled = list(sampled_correct) + list(sampled_incorrect)
        labels      = ["Correct"] * len(sampled_correct) + ["Incorrect"] * len(sampled_incorrect)
        n = len(all_sampled)
        fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
        if n == 1: axes = axes.reshape(2, 1)
        fig.suptitle(f"Attention Map — {cls_name}", fontsize=14)
        for col, (img_idx, lbl) in enumerate(zip(all_sampled, labels)):
            img_path = test_paths[img_idx]
            if img_path.lower().endswith(('.tif', '.tiff', '.ome.tif')):
                import tifffile, warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    img_arr = tifffile.imread(img_path)
                if img_arr.ndim == 4: img_arr = img_arr[0, 0]
                elif img_arr.ndim == 3 and img_arr.shape[0] <= 4: img_arr = img_arr[0]
                if img_arr.dtype != np.uint8:
                    img_arr = (255 * (img_arr.astype(np.float32) / img_arr.max())).astype(np.uint8)
                if img_arr.ndim == 2: img_arr = np.stack([img_arr]*3, axis=-1)
                img_pil = Image.fromarray(img_arr)
            else:
                img_pil = Image.open(img_path).convert("RGB")
            img_tensor = preprocess_val(img_pil).unsqueeze(0)
            heatmap    = attn_map.generate(img_tensor, model)
            img_np     = np.array(img_pil.resize((224, 224)))
            heatmap_uint8 = np.uint8(255 * heatmap)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            overlay = np.uint8(0.4 * img_np + 0.6 * heatmap_color)
            axes[0, col].imshow(img_np)
            axes[0, col].set_title(f"Original ({lbl})", fontsize=9)
            axes[0, col].axis("off")
            axes[1, col].imshow(overlay)
            axes[1, col].set_title("Attn Map", fontsize=9)
            axes[1, col].axis("off")
        plt.tight_layout()
        fname = f"{prefix}_attn_map_{cls_name}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
        wandb_imgs[f"attention_map_{cls_name}"] = wandb.Image(fname)
    return wandb_imgs


class SingleScaleHead(nn.Module):
    """Frozen BiomedCLIP backbone, single-scale encode_image output, projection head."""
    def __init__(self, visual_model, embed_dim=512):
        super().__init__()
        self.visual     = visual_model
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        ).to(DEVICE)
        for param in self.visual.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            feat = self.visual(x)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        out  = self.projection(feat)
        return out / out.norm(dim=-1, keepdim=True)


train_paths, train_labels, test_paths, test_labels = split_dataset(
    DATA_DIR, total_samples=None, train_ratio=0.8)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_paths, train_labels, test_size=0.125, random_state=42)

train_labels_idx = np.array([np.argmax(l) for l in train_labels])
val_labels_idx   = np.array([np.argmax(l) for l in val_labels])
test_labels_idx  = np.array([np.argmax(l) for l in test_labels])

class_counts  = np.bincount(train_labels_idx, minlength=NUM_CLASSES).astype(np.float32)
class_weights = torch.tensor(1.0 / (class_counts + 1e-8), dtype=torch.float32).to(DEVICE)
class_weights = class_weights / class_weights.sum() * NUM_CLASSES

train_loader = DataLoader(CellDataset(train_paths, train_labels_idx, preprocess_train),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(CellDataset(val_paths, val_labels_idx, preprocess_val),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(CellDataset(test_paths, test_labels_idx, preprocess_val),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

fused_model = SingleScaleHead(clip_model.visual)

wandb.init(
    project="BIOCE",
    name=f"{PREFIX}_frozen_single_scale",
    config={
        "model": MODEL_NAME, "epochs": NUM_EPOCHS, "patience": PATIENCE,
        "lr_head": LR_HEAD, "batch_size": BATCH_SIZE,
        "class_weighted_loss": True, "scheduler": "CosineAnnealingLR",
        "backbone": "frozen", "feature_extraction": "single_scale_encode_image",
        "step": f"{PREFIX}_frozen_single_scale", "prompt_template": PROMPT_TEMPLATE,
    }
)

optimizer        = optim.AdamW(fused_model.projection.parameters(), lr=LR_HEAD, weight_decay=0.01)
scheduler        = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
criterion        = nn.CrossEntropyLoss(weight=class_weights)
logit_scale      = clip_model.logit_scale
best_val_bacc    = 0.0
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    fused_model.train()
    running_loss = 0.0
    for batch_images, batch_labels in train_loader:
        batch_images = batch_images.to(DEVICE)
        batch_labels = batch_labels.to(DEVICE)
        optimizer.zero_grad()
        img_features = fused_model(batch_images)
        logits       = logit_scale.exp() * (img_features @ text_emb.t())
        loss         = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_images.size(0)
    scheduler.step()
    epoch_train_loss = running_loss / len(train_loader.dataset)

    fused_model.eval()
    val_probs_list, val_true_list, val_running_loss = [], [], 0.0
    with torch.no_grad():
        for batch_images, batch_labels in val_loader:
            batch_images = batch_images.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            img_features = fused_model(batch_images)
            logits       = logit_scale.exp() * (img_features @ text_emb.t())
            val_running_loss += criterion(logits, batch_labels).item() * batch_images.size(0)
            val_probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
            val_true_list.append(batch_labels.cpu().numpy())

    val_probs = np.vstack(val_probs_list)
    val_true  = np.concatenate(val_true_list)
    val_preds = np.argmax(val_probs, axis=1)
    bal_acc   = balanced_accuracy_score(val_true, val_preds)
    macro_auc = roc_auc_score(np.eye(NUM_CLASSES)[val_true], val_probs, average="macro")

    wandb.log({"epoch": epoch+1, "train_loss": epoch_train_loss,
               "val_loss": val_running_loss/len(val_loader.dataset),
               "val_balanced_acc": bal_acc, "macro_auc": macro_auc,
               "lr_head": optimizer.param_groups[0]["lr"]})

    if bal_acc > best_val_bacc:
        best_val_bacc = bal_acc
        patience_counter = 0
        torch.save(fused_model.projection.state_dict(), f"{PREFIX}_best.pt")
    else:
        patience_counter += 1

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {epoch_train_loss:.4f} | "
          f"Val Loss: {val_running_loss/len(val_loader.dataset):.4f} | "
          f"Val BACC: {bal_acc:.4f} | AUC: {macro_auc:.4f} | Patience: {patience_counter}/{PATIENCE}")

    if patience_counter >= PATIENCE:
        print(f"Early stopping at epoch {epoch+1}")
        break

fused_model.projection.load_state_dict(torch.load(f"{PREFIX}_best.pt"))
fused_model.eval()

test_probs_list, test_true_list = [], []
with torch.no_grad():
    for batch_images, batch_labels in test_loader:
        img_features = fused_model(batch_images.to(DEVICE))
        logits       = logit_scale.exp() * (img_features @ text_emb.t())
        test_probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
        test_true_list.append(batch_labels.cpu().numpy())

y_score = np.vstack(test_probs_list)
y_true  = np.concatenate(test_true_list)
y_pred  = np.argmax(y_score, axis=1)

cm = confusion_matrix(y_true, y_pred)
cm_percent = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True) * 100
plt.figure(figsize=(8, 6))
plt.imshow(cm_percent, interpolation="nearest", cmap=plt.cm.Blues)
plt.title(f"{PREFIX} Frozen Single-Scale — Confusion Matrix (%)")
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
plt.title(f"{PREFIX} Frozen Single-Scale ROC Curves")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{PREFIX}_roc.png", dpi=300)
plt.close()

final_macro_auc = roc_auc_score(np.eye(NUM_CLASSES)[y_true], y_score, average="macro")
final_bal_acc   = balanced_accuracy_score(y_true, y_pred)
final_wacc      = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
with open(f"{PREFIX}_report.json", "w") as f:
    json.dump(report, f, indent=2)

per_class_auc  = {"auc_" + k: v for k, v in auc_scores.items()}
per_class_f1   = {"f1_" + c: report[c]["f1-score"] for c in CLASS_NAMES}
per_class_prec = {"precision_" + c: report[c]["precision"] for c in CLASS_NAMES}
per_class_rec  = {"recall_" + c: report[c]["recall"] for c in CLASS_NAMES}

torch.cuda.empty_cache()
print("\nGenerating attention map visualizations...")
attn_imgs = generate_attention_grid(clip_model, test_paths, y_true, y_pred,
                                     CLASS_NAMES, DEVICE, PREFIX)
wandb.log({
    "final_balanced_accuracy": final_bal_acc,
    "final_macro_auc": final_macro_auc,
    "final_weighted_accuracy": final_wacc,
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
    **attn_imgs,
})
wandb.finish()
print(f"\nTest Balanced Accuracy : {final_bal_acc:.4f}")
print(f"Test Macro AUC         : {final_macro_auc:.4f}")
