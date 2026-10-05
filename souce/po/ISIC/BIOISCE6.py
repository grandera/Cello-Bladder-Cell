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

class SoftPrompt(nn.Module):
    """Class-Specific Context (CSC) CoOp for BiomedCLIP/PubMedBERT.
    
    Learns N_CTX context tokens per class in BERT embedding space (768-dim).
    These replace the first N_CTX tokens after BOS in the prompt sequence.
    The full sequence is then passed through the frozen text encoder.
    """
    def __init__(self, clip_model, tokenizer, class_names, n_ctx,
                 prompt_template, device, ctx_len=256):
        super().__init__()
        self.clip_model  = clip_model
        self.tokenizer   = tokenizer
        self.class_names = class_names
        self.n_ctx       = n_ctx
        self.device      = device
        self.ctx_len     = ctx_len

        word_emb = clip_model.text.transformer.embeddings.word_embeddings

        with torch.no_grad():
            prompts    = [prompt_template.format(c) for c in class_names]
            tokens     = tokenizer(prompts, context_length=ctx_len).to(device)
            # Get full token embeddings: [num_classes, ctx_len, 768]
            embeddings = word_emb(tokens)

        # Initialise CSC context from actual prompt tokens (positions 1..n_ctx)
        # Shape: [num_classes, n_ctx, 768]
        ctx_init = embeddings[:, 1:n_ctx + 1, :].clone().detach()
        self.ctx = nn.Parameter(ctx_init)

        # Store full token ids for positional and type embeddings
        self.register_buffer("token_ids", tokens)  # [num_classes, ctx_len]

    def forward(self):
        """Compute text embeddings with learned context tokens."""
        word_emb    = self.clip_model.text.transformer.embeddings.word_embeddings
        text_feats  = []

        for i in range(len(self.class_names)):
            token_ids_i = self.token_ids[i]           # [ctx_len]

            # Full word embeddings for this class [ctx_len, 768]
            with torch.no_grad():
                full_emb = word_emb(token_ids_i.unsqueeze(0))  # [1, ctx_len, 768]

            # Replace positions 1..n_ctx with learned context
            prefix = full_emb[:, :1, :]                        # BOS [1, 1, 768]
            ctx_i  = self.ctx[i:i+1, :, :]                     # [1, n_ctx, 768]
            suffix = full_emb[:, 1 + self.n_ctx:, :]           # rest [1, ctx_len-n_ctx-1, 768]
            new_emb = torch.cat([prefix, ctx_i, suffix], dim=1) # [1, ctx_len, 768]

            # Pass through BERT with modified embeddings
            # We hook into the transformer by replacing word embeddings output
            # Use a temporary hook
            hook_output = {}
            def make_hook(storage):
                def hook(module, input, output):
                    storage['out'] = new_emb
                    return new_emb
                return hook

            handle = word_emb.register_forward_hook(make_hook(hook_output))
            try:
                feat = self.clip_model.encode_text(token_ids_i.unsqueeze(0))
            finally:
                handle.remove()

            feat = feat / feat.norm(dim=-1, keepdim=True)
            text_feats.append(feat)

        return torch.cat(text_feats, dim=0)  # [num_classes, 512]


from PIL import Image

import pandas as pd
from torch.utils.data import Dataset

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

PREFIX          = "BIOISCE6"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR        = "ISIC2019"
CLASS_NAMES     = [
    "melanoma", "melanocytic_nevus", "basal_cell_carcinoma",
    "actinic_keratosis", "benign_keratosis",
    "dermatofibroma", "vascular_lesion", "squamous_cell_carcinoma"
]
NUM_CLASSES     = len(CLASS_NAMES)
MODEL_NAME      = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
BATCH_SIZE      = 32
NUM_EPOCHS      = 100
PATIENCE        = 10
LR_HEAD         = 1e-4
LR_BACKBONE     = 1e-5
LAYER_DECAY     = 0.80
LR_TEMP         = 1e-6
N_CTX           = 16
LR_PROMPT       = 3e-4
PROMPT_TEMPLATE = "a dermoscopy image of {}"

clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(MODEL_NAME)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)
clip_model.to(DEVICE)

soft_prompt = SoftPrompt(
    clip_model      = clip_model,
    tokenizer       = tokenizer,
    class_names     = CLASS_NAMES,
    n_ctx           = N_CTX,
    prompt_template = PROMPT_TEMPLATE,
    device          = DEVICE,
    ctx_len         = 256,
).to(DEVICE)




class ISICDataset(Dataset):
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


class ClipFusionHead(nn.Module):
    """Frozen BiomedCLIP backbone, multi-scale CLS fusion blocks 3,7,11. timm ViT format."""
    def __init__(self, visual_model, embed_dim=512):
        super().__init__()
        self.visual     = visual_model
        self.projection = nn.Sequential(
            nn.Linear(768 * 3, 768),
            nn.ReLU(),
            nn.Linear(768, embed_dim)
        ).to(DEVICE)
        for param in self.visual.parameters():
            param.requires_grad = False
        for block in self.visual.trunk.blocks[-6:]:
            for param in block.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.visual.trunk.patch_embed(x)
        x = self.visual.trunk._pos_embed(x)
        x = self.visual.trunk.norm_pre(x)
        cls_tokens = []
        for i, block in enumerate(self.visual.trunk.blocks):
            x = block(x)
            if i in [3, 7, 11]:
                cls_tokens.append(x[:, 0, :])
        fused = torch.cat(cls_tokens, dim=-1)
        out   = self.projection(fused)
        return out / out.norm(dim=-1, keepdim=True)


df            = pd.read_csv(os.path.join(DATA_DIR, "isic2019_labels.csv"))
base_dir      = os.path.join(DATA_DIR, "images", "ISIC_2019_Training_Input")
image_paths   = [os.path.join(base_dir, p.split("/")[-1].replace(".jpg", "") + ".jpg")
                 for p in df["image_path"].tolist()]
labels_str    = df["label"].tolist()
label_indices = np.array([CLASS_NAMES.index(l) for l in labels_str])

train_paths, test_paths, train_labels_idx, test_labels_idx = train_test_split(
    image_paths, label_indices, test_size=0.20,
    stratify=label_indices, random_state=42)
train_paths, val_paths, train_labels_idx, val_labels_idx = train_test_split(
    train_paths, train_labels_idx, test_size=0.125,
    stratify=train_labels_idx, random_state=42)

print(f"Train: {len(train_paths)}  Val: {len(val_paths)}  Test: {len(test_paths)}")

class_counts  = np.bincount(train_labels_idx, minlength=NUM_CLASSES).astype(np.float32)
class_weights = torch.tensor(1.0 / (class_counts + 1e-8), dtype=torch.float32).to(DEVICE)
class_weights = class_weights / class_weights.sum() * NUM_CLASSES

train_loader = DataLoader(ISICDataset(train_paths, train_labels_idx, preprocess_train),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(ISICDataset(val_paths, val_labels_idx, preprocess_val),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(ISICDataset(test_paths, test_labels_idx, preprocess_val),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

fused_model = ClipFusionHead(clip_model.visual)

wandb.init(
    project="BIOIS",
    name=f"{PREFIX}_coop_csc",
    config={
        "model": MODEL_NAME, "epochs": NUM_EPOCHS, "patience": PATIENCE,
        "lr_head": LR_HEAD, "lr_backbone": LR_BACKBONE, "lr_temp": LR_TEMP, "lr_prompt": LR_PROMPT, "n_ctx": N_CTX, "batch_size": BATCH_SIZE,
        "class_weighted_loss": True, "scheduler": "CosineAnnealingLR",
        "backbone": "unfrozen_last6_learnable_logit_scale_coop_csc", "feature_extraction": "multiscale_cls_blocks_3_7_11",
        "step": f"{PREFIX}_coop_csc", "prompt_template": PROMPT_TEMPLATE,
    }
)

clip_model.logit_scale.requires_grad = True
optimizer = optim.AdamW([
    {"params": fused_model.projection.parameters(), "lr": LR_HEAD},
    # LLRD: block 11 gets LR_BACKBONE, each lower block multiplied by LAYER_DECAY
    *[{"params": fused_model.visual.trunk.blocks[-(i+1)].parameters(),
       "lr": LR_BACKBONE * (LAYER_DECAY ** i)} for i in range(6)],
    {"params": [clip_model.logit_scale], "lr": LR_TEMP},
    {"params": [soft_prompt.ctx], "lr": LR_PROMPT},
], weight_decay=0.01)
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
        text_emb     = soft_prompt()
        logits       = logit_scale.exp() * (img_features @ text_emb.t())
        loss         = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            clip_model.logit_scale.clamp_(0, 4.6052)
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
            text_emb     = soft_prompt()
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
               "lr_head": optimizer.param_groups[0]["lr"],
               "lr_backbone": optimizer.param_groups[1]["lr"],
               "lr_temp": optimizer.param_groups[7]["lr"],  # logit scale group
               "lr_prompt": optimizer.param_groups[3]["lr"],
               "logit_scale_value": logit_scale.exp().item()})

    if bal_acc > best_val_bacc:
        best_val_bacc = bal_acc
        patience_counter = 0
        torch.save({
            "projection": fused_model.projection.state_dict(),
            "unfrozen_blocks": [b.state_dict() for b in fused_model.visual.trunk.blocks[-6:]],
            "logit_scale": clip_model.logit_scale.data,
            "soft_prompt": soft_prompt.ctx.data,
        }, f"{PREFIX}_best.pt")
    else:
        patience_counter += 1

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {epoch_train_loss:.4f} | "
          f"Val Loss: {val_running_loss/len(val_loader.dataset):.4f} | "
          f"Val BACC: {bal_acc:.4f} | AUC: {macro_auc:.4f} | Patience: {patience_counter}/{PATIENCE}")

    if patience_counter >= PATIENCE:
        print(f"Early stopping at epoch {epoch+1}")
        break

checkpoint = torch.load(f"{PREFIX}_best.pt")
fused_model.projection.load_state_dict(checkpoint["projection"])
for b, state in zip(fused_model.visual.trunk.blocks[-6:], checkpoint["unfrozen_blocks"]):
    b.load_state_dict(state)
clip_model.logit_scale.data = checkpoint["logit_scale"]
soft_prompt.ctx.data        = checkpoint["soft_prompt"]
fused_model.eval()

test_probs_list, test_true_list = [], []
with torch.no_grad():
    for batch_images, batch_labels in test_loader:
        img_features = fused_model(batch_images.to(DEVICE))
        text_emb     = soft_prompt()
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
plt.title(f"{PREFIX} Unfrozen Multi-Scale — Confusion Matrix (%)")
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
plt.title(f"{PREFIX} Unfrozen Multi-Scale ROC Curves")
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
attn_imgs = generate_attention_grid(fused_model, test_paths, y_true, y_pred,
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
    "final_logit_scale": logit_scale.exp().item(),
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
