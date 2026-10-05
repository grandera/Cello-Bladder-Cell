import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import wandb
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import open_clip
from transformers import CLIPModel, CLIPProcessor
from torchvision import transforms

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_DIR   = "/raid/DATASETS/po/Raabin/Train"
TEST_DIR    = "/raid/DATASETS/po/Raabin/Test-A"
CLASS_NAMES = ["Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]
NUM_CLASSES     = len(CLASS_NAMES)
PROMPT_TEMPLATE = "a microscopy image of {} white blood cell"
BATCH_SIZE      = 32

BACKBONE_CONFIGS = [
    {"name": "BiomedCLIP",       "loader": "open_clip",    "model_name": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"},
    {"name": "PubMedCLIP",       "loader": "transformers", "model_name": "flaviagiammarino/pubmed-clip-vit-base-patch32"},
    {"name": "PLIP",             "loader": "transformers", "model_name": "vinid/plip"},
    {"name": "OpenCLIP_ViT_B16", "loader": "open_clip",    "model_name": "ViT-B-16",  "pretrained": "openai"},
    {"name": "OpenCLIP_ViT_L14", "loader": "open_clip",    "model_name": "ViT-L-14",  "pretrained": "openai"},
    {"name": "OpenCLIP_ViT_H14", "loader": "open_clip",    "model_name": "ViT-H-14",  "pretrained": "laion2b_s32b_b79k"},
]

PUBMED_PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])


class WBCDataset(Dataset):
    def __init__(self, image_paths, label_indices, transform=None):
        self.image_paths   = image_paths
        self.label_indices = label_indices
        self.transform     = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.label_indices[idx]


def load_split(data_dir):
    paths, labels = [], []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.exists(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                paths.append(os.path.join(cls_dir, fname))
                labels.append(cls_idx)
    return paths, np.array(labels)


def load_model(cfg):
    if cfg["loader"] == "open_clip":
        if "pretrained" in cfg:
            model, _, preprocess_val = open_clip.create_model_and_transforms(
                cfg["model_name"], pretrained=cfg["pretrained"])
        else:
            model, _, preprocess_val = open_clip.create_model_and_transforms(cfg["model_name"])
        tokenizer = open_clip.get_tokenizer(cfg["model_name"])
        model.to(DEVICE).eval()

        def encode_image(imgs): return model.encode_image(imgs.to(DEVICE))
        def encode_text(prompts): return model.encode_text(tokenizer(prompts).to(DEVICE))
        def logit_scale(): return model.logit_scale.exp()

        return encode_image, encode_text, logit_scale, preprocess_val, model

    else:
        hf_model  = CLIPModel.from_pretrained(cfg["model_name"]).to(DEVICE)
        processor = CLIPProcessor.from_pretrained(cfg["model_name"])
        hf_model.eval()

        def encode_image(imgs): return hf_model.get_image_features(pixel_values=imgs.to(DEVICE))
        def encode_text(prompts):
            enc = processor(text=prompts, return_tensors="pt", padding=True,
                            truncation=True, max_length=77)
            return hf_model.get_text_features(**{k: v.to(DEVICE) for k, v in enc.items()})
        def logit_scale(): return hf_model.logit_scale.exp()

        return encode_image, encode_text, logit_scale, PUBMED_PREPROCESS, hf_model


def run_zero_shot(encode_image_fn, encode_text_fn, logit_scale_fn, test_loader):
    prompts = [PROMPT_TEMPLATE.format(c.lower()) for c in CLASS_NAMES]
    with torch.no_grad():
        text_emb = encode_text_fn(prompts)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    all_scores, all_true = [], []
    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            img_emb = encode_image_fn(batch_images)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            probs   = torch.softmax(logit_scale_fn() * (img_emb @ text_emb.t()), dim=1)
            all_scores.append(probs.cpu().numpy())
            all_true.append(batch_labels.numpy())
    y_score = np.vstack(all_scores)
    y_true  = np.concatenate(all_true)
    y_pred  = np.argmax(y_score, axis=1)
    return balanced_accuracy_score(y_true, y_pred), \
           roc_auc_score(np.eye(NUM_CLASSES)[y_true], y_score, average="macro")


def plot_summary(summary_rows, prefix):
    models = [r[0] for r in summary_rows]
    x      = np.arange(len(models))
    w      = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(x, [r[1] for r in summary_rows], w, color="steelblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=20, ha="right")
    axes[0].set_ylabel("Balanced Accuracy")
    axes[0].set_title(f"{prefix} Zero-Shot Balanced Accuracy")
    axes[0].set_ylim(0, 1)
    axes[1].bar(x, [r[2] for r in summary_rows], w, color="darkorange")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=20, ha="right")
    axes[1].set_ylabel("Macro AUC")
    axes[1].set_title(f"{prefix} Zero-Shot Macro AUC")
    axes[1].set_ylim(0, 1)
    plt.tight_layout()
    fname = f"{prefix}_backbone_zeroshot.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    return fname


# ── Data ──────────────────────────────────────────────────────────────────────
test_paths, test_labels = load_split(TEST_DIR)

# ── Run ───────────────────────────────────────────────────────────────────────
wandb.init(
    project="backbone_comparison",
    name="WB_backbone_zeroshot",
    config={
        "dataset": "Raabin-WBC",
        "num_classes": NUM_CLASSES,
        "batch_size": BATCH_SIZE,
        "models_tested": [c["name"] for c in BACKBONE_CONFIGS],
        "evaluation": "zero_shot",
        "test_set": "Test-A",
        "prompt_template": PROMPT_TEMPLATE,
    }
)

summary_rows = []

for cfg in BACKBONE_CONFIGS:
    print(f"\n{'='*60}")
    print(f"Testing backbone: {cfg['name']}")
    print(f"{'='*60}")

    encode_image_fn, encode_text_fn, logit_scale_fn, preprocess_val, raw_model = load_model(cfg)

    test_loader = DataLoader(WBCDataset(test_paths, test_labels, preprocess_val),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    bacc, auc = run_zero_shot(encode_image_fn, encode_text_fn, logit_scale_fn, test_loader)

    wandb.log({f"WB_{cfg['name']}_bacc": bacc, f"WB_{cfg['name']}_auc": auc})
    print(f"  BACC: {bacc:.4f} | AUC: {auc:.4f}")
    summary_rows.append([cfg["name"], bacc, auc])

    del raw_model
    torch.cuda.empty_cache()

summary_fname = plot_summary(summary_rows, "WB")
wandb.log({
    "WB_zeroshot_summary_chart": wandb.Image(summary_fname),
    "WB_zeroshot_summary": wandb.Table(
        columns=["model", "balanced_accuracy", "macro_auc"],
        data=summary_rows
    )
})

print("\n\n===== RAABIN-WBC ZERO-SHOT BACKBONE COMPARISON =====")
print(f"{'Model':<25} {'BACC':>10} {'Macro AUC':>10}")
print("-" * 48)
for row in summary_rows:
    print(f"{row[0]:<25} {row[1]:>10.4f} {row[2]:>10.4f}")

wandb.finish()
