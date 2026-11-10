import sys
import os
#sys.stderr = open(os.devnull, "w")

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="TiffTag 266: 0 is not a valid FILLORDER")
warnings.filterwarnings("ignore", message=".*TIFFReadDirectory.*")
import torch
from PIL import Image
from torch.utils.data import Dataset
import tifffile
tifffile.TIFF.ERRORLEVEL = 0
import numpy as np
import random

cell_to_labels = {
    "SV-HUC-1": ["NA"],
    "RT-112": ["Ta"],
    "HT-1376": ["T3"],
    "5637": ["T2"],
    "253J-BV": ["T4"],
    "TCC-SUP": ["T4"],
    "T24": ["T3"],
    "J82": ["T3"],
    "639-V": ["T3"],
    "HT-1197": ["T2"],
    "CAL-29": ["T2"],
    "BFTC-905": ["T3", "T4"],
    "UM-HUC-3": ["T2", "T4"],
    "BFTC-909": ["T3", "T4"],
    "RT4": ["T2"]
}
_raw_labels = {l for v in cell_to_labels.values() for l in v if l != "T1"}
_clinical_order = ["NA", "Ta", "T2", "T3", "T4"]
all_labels = [l for l in _clinical_order if l in _raw_labels]
label_to_index = {label: i for i, label in enumerate(all_labels)}

AMBIGUOUS_FOLDERS = {"BFTC-905", "UM-HUC-3", "BFTC-909"}

btc_classes = ['HGC', 'LGC', 'NST', 'NTL']
class_to_idx = {cls: i for i, cls in enumerate(btc_classes)}

def get_btc_image_paths_and_labels(root):
    image_paths = []
    labels = []
    for cls in btc_classes:
        cls_dir = os.path.join(root, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(cls_dir, fname))
                labels.append(class_to_idx[cls])
    return image_paths, labels

class BTCImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=int)  
        return img, label

def is_relevant_folder(folder_name):
    return any(sub in folder_name for sub in cell_to_labels.keys())

def split_dataset(data_dir, total_samples=None, train_ratio=0.7, seed=42):
    import os
    import random
    import numpy as np

    all_samples = []
    for root, dirs, files in os.walk(data_dir):
        folder_name = os.path.basename(root)
        parent_folder = os.path.basename(os.path.dirname(root))
        if folder_name.lower().endswith("_single") and is_relevant_folder(parent_folder):
            matched_cell = next((sub for sub in cell_to_labels.keys() if sub in parent_folder), None)
            if matched_cell is None:
                continue
            is_ambiguous = matched_cell in AMBIGUOUS_FOLDERS
            for fname in files:
                if fname.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')) and 'Ch1' in fname:
                    fpath = os.path.join(root, fname)
                    all_samples.append((fpath, matched_cell, is_ambiguous))

    random.seed(seed)
    non_ambiguous = [(p, f) for p, f, is_amb in all_samples if not is_amb]
    random.shuffle(non_ambiguous)

    if total_samples is None:
        total_samples = len(non_ambiguous)
    n_train = int(total_samples * train_ratio)
    n_test = total_samples - n_train

    train_samples = non_ambiguous[:n_train]
    test_samples = non_ambiguous[n_train:n_train + n_test]

    label_to_nonamb_folders = {label: [] for label in all_labels}
    for _, folder, is_amb in all_samples:
        if not is_amb:
            for label in cell_to_labels.get(folder, []):
                label_to_nonamb_folders[label].append(folder)
    guaranteed_train = []
    used_train_folders = set()
    for label in all_labels:
        folders = label_to_nonamb_folders[label]
        if folders:
            for sample in train_samples:
                if sample[1] == folders[0] and sample not in guaranteed_train:
                    guaranteed_train.append(sample)
                    used_train_folders.add(sample[0])
                    break
    train_samples_remaining = [s for s in train_samples if s[0] not in used_train_folders]
    train_samples = guaranteed_train + train_samples_remaining[:n_train - len(guaranteed_train)]

    label_to_any_folders = {label: [] for label in all_labels}
    for _, folder, is_amb in all_samples:
        if not is_amb:
            for label in cell_to_labels.get(folder, []):
                label_to_any_folders[label].append(folder)
    guaranteed_test = []
    used_test_folders = set()
    for label in all_labels:
        folders = label_to_any_folders[label]
        if folders:
            for sample in test_samples:
                if sample[1] == folders[0] and sample not in guaranteed_test:
                    guaranteed_test.append(sample)
                    used_test_folders.add(sample[0])
                    break
    test_samples_remaining = [s for s in test_samples if s[0] not in used_test_folders]
    test_samples = guaranteed_test + test_samples_remaining[:n_test - len(guaranteed_test)]
    random.shuffle(test_samples)

    train_label_counts = {label: 0 for label in all_labels}
    test_label_counts = {label: 0 for label in all_labels}

    for _, folder in train_samples:
        for label in cell_to_labels.get(folder, []):
            train_label_counts[label] += 1
    for _, folder in test_samples:
        for label in cell_to_labels.get(folder, []):
            test_label_counts[label] += 1

    print(f"Train samples: {len(train_samples)}")
    for label in all_labels:
        print(f"{label}: {train_label_counts[label]}")
    print(f"Test samples: {len(test_samples)}")
    for label in all_labels:    
        print(f"{label}: {test_label_counts[label]}")

    def get_labels(folder):
        label_vec = np.zeros(len(all_labels), dtype=np.float32)
        for l in cell_to_labels.get(folder, []):
            if l in label_to_index:
                label_vec[label_to_index[l]] = 1.0
        return label_vec

    train_paths = [p for p, _ in train_samples]
    train_labels = [get_labels(f) for _, f in train_samples]
    test_paths = [p for p, f in test_samples]
    test_labels = [get_labels(f) for _, f in test_samples]

    return train_paths, train_labels, test_paths, test_labels

class CellDataset(Dataset):
    def __init__(self, samples, labels, transform=None):
        self.samples = samples
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if img_path.lower().endswith('.ome.tif'):
                img = tifffile.imread(img_path)
                if img.ndim == 4:
                    img = img[0, 0]
                elif img.ndim == 3:
                    if img.shape[0] <= 4:
                        img = img[0]
                    elif img.shape[2] <= 4:
                        pass
                if img.dtype != np.uint8:
                    img = (255 * (img.astype(np.float32) / img.max())).astype(np.uint8)
                if img.ndim == 2:
                    img = np.stack([img]*3, axis=-1)
                elif img.ndim == 3 and img.shape[2] == 1:
                    img = np.repeat(img, 3, axis=2)
                img = Image.fromarray(img)
            else:
                img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

if __name__ == "__main__":
    print("split_dataset:", 'split_dataset' in dir())
    print("CellDataset:", 'CellDataset' in dir())
    print("all_labels:", 'all_labels' in dir())