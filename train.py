import argparse
import json
import os
import random
from typing import List, Tuple, Dict

import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import StratifiedGroupKFold


class SkinDefectsImageDataset(Dataset):

	def __init__(self, items: List[Tuple[str, int]], transform: transforms.Compose) -> None:
		self.items = items
		self.transform = transform

	def __len__(self) -> int:
		return len(self.items)

	def __getitem__(self, idx: int):
		image_path, label = self.items[idx]
		image = Image.open(image_path).convert("RGB")
		image = self.transform(image)
		return image, label


def expand_csv_rows_to_images(csv_path: str, files_root: str) -> Tuple[List[Tuple[str, str, str]], List[str]]:

	# Returns list of (absolute_image_path, class_name, group_id) and unique class names list
	df = pd.read_csv(csv_path)
	required_cols = {"id", "front", "left_side", "right_side", "type"}
	missing = required_cols.difference(df.columns)
	if missing:
		raise ValueError(f"CSV missing required columns: {missing}")

	items: List[Tuple[str, str, str]] = []
	for _, row in df.iterrows():
		label = str(row["type"]).strip()
		group_id = str(row["id"]).strip()
		for col in ("front", "left_side", "right_side"):
			rel_path = str(row[col]).lstrip("/").replace("/", os.sep)
			abs_path = os.path.join(files_root, rel_path)
			if os.path.isfile(abs_path):
				items.append((abs_path, label, group_id))

	class_names = sorted({label for _, label, _ in items})
	return items, class_names


def load_folder_dataset(data_dir: str) -> Tuple[List[Tuple[str, str, str]], List[str]]:
	"""Load images from folder structure where each subfolder is a class.
	
	Expected structure:
		data_dir/
			ClassName1/
				image1.jpg
				image2.jpg
			ClassName2/
				image3.jpg
	
	Returns list of (absolute_image_path, class_name, group_id) and unique class names list.
	group_id is set to image filename for folder-based datasets.
	"""
	items: List[Tuple[str, str, str]] = []
	valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
	
	if not os.path.isdir(data_dir):
		raise ValueError(f"Data directory not found: {data_dir}")
	
	class_names = sorted([d for d in os.listdir(data_dir) 
						  if os.path.isdir(os.path.join(data_dir, d))])
	
	for class_name in class_names:
		class_dir = os.path.join(data_dir, class_name)
		for filename in os.listdir(class_dir):
			ext = os.path.splitext(filename)[1].lower()
			if ext in valid_extensions:
				abs_path = os.path.join(class_dir, filename)
				# Use filename as group_id (each image is independent)
				group_id = os.path.splitext(filename)[0]
				items.append((abs_path, class_name, group_id))
	
	return items, class_names


def build_dataloaders(
	items: List[Tuple[str, str, str]],
	class_to_idx: Dict[str, int],
	image_size: int,
	batch_size: int,
	val_size: float,
	seed: int,
) -> Tuple[DataLoader, DataLoader]:

	# Group-aware, stratified split so that different views of the same subject stay in the same fold
	labels = [class_to_idx[label] for _, label, _ in items]
	groups = [group for _, _, group in items]

	n_splits = int(1 / max(1e-6, val_size))
	n_splits = max(2, min(5, n_splits))
	sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
	train_idx, val_idx = next(sgkf.split(X=items, y=labels, groups=groups))
	train_items = [items[i] for i in train_idx]
	val_items = [items[i] for i in val_idx]

	train_tfms = transforms.Compose([
		transforms.Resize((image_size, image_size)),
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.RandomRotation(degrees=8),
		# Keep color jitter modest and avoid hue shifts to preserve redness cues
		transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	val_tfms = transforms.Compose([
		transforms.Resize((image_size, image_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	def encode(items_list: List[Tuple[str, str, str]]):
		return [(path, class_to_idx[label]) for path, label, _ in items_list]

	train_ds = SkinDefectsImageDataset(encode(train_items), train_tfms)
	val_ds = SkinDefectsImageDataset(encode(val_items), val_tfms)

	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
	val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
	return train_loader, val_loader


def create_model(arch: str, num_classes: int, freeze_backbone: bool = False) -> nn.Module:

	arch = arch.lower()
	if arch == "resnet18":
		model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
		if freeze_backbone:
			for param in model.parameters():
				param.requires_grad = False
			# keep fc trainable
		in_features = model.fc.in_features
		model.fc = nn.Linear(in_features, num_classes)
		return model
	elif arch == "efficientnet_b0":
		model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
		if freeze_backbone:
			for param in model.parameters():
				param.requires_grad = False
		# Replace classifier
		if isinstance(model.classifier, nn.Sequential):
			in_features = model.classifier[-1].in_features
			model.classifier[-1] = nn.Linear(in_features, num_classes)
		else:
			raise RuntimeError("Unexpected EfficientNet classifier structure")
		return model
	elif arch == "convnext_tiny":
		model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
		if freeze_backbone:
			for param in model.parameters():
				param.requires_grad = False
		# Replace classifier head
		in_features = model.classifier[-1].in_features
		model.classifier[-1] = nn.Linear(in_features, num_classes)
		return model
	else:
		raise ValueError(f"Unknown arch: {arch}")


def train_one_epoch(model, loader, criterion, optimizer, device) -> Tuple[float, float]:

	model.train()
	running_loss = 0.0
	running_corrects = 0
	total = 0

	use_mixup = getattr(args_holder, 'mixup', 0.0) and getattr(args_holder, 'mixup', 0.0) > 0.0
	use_cutmix = getattr(args_holder, 'cutmix', 0.0) and getattr(args_holder, 'cutmix', 0.0) > 0.0

	def rand_bbox(size, lam):
		W = size[2]
		H = size[3]
		import numpy as np
		cut_rat = np.sqrt(1. - lam)
		cut_w = int(W * cut_rat)
		cut_h = int(H * cut_rat)
		cx = np.random.randint(W)
		cy = np.random.randint(H)
		x1 = np.clip(cx - cut_w // 2, 0, W)
		y1 = np.clip(cy - cut_h // 2, 0, H)
		x2 = np.clip(cx + cut_w // 2, 0, W)
		y2 = np.clip(cy + cut_h // 2, 0, H)
		return x1, y1, x2, y2

	for images, labels in loader:
		images = images.to(device)
		labels = labels.to(device)

		optimizer.zero_grad()

		mixed = False
		if use_mixup:
			lam = torch.distributions.Beta(args_holder.mixup, args_holder.mixup).sample().item()
			index = torch.randperm(images.size(0), device=images.device)
			images = lam * images + (1 - lam) * images[index, :]
			outputs = model(images)
			loss = lam * criterion(outputs, labels) + (1 - lam) * criterion(outputs, labels[index])
			mixed = True
		elif use_cutmix:
			lam = torch.distributions.Beta(args_holder.cutmix, args_holder.cutmix).sample().item()
			index = torch.randperm(images.size(0), device=images.device)
			x1, y1, x2, y2 = rand_bbox(images.size(), lam)
			images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
			outputs = model(images)
			lam_adj = 1 - ((x2 - x1) * (y2 - y1) / (images.size(2) * images.size(3)))
			loss = lam_adj * criterion(outputs, labels) + (1 - lam_adj) * criterion(outputs, labels[index])
			mixed = True

		if not mixed:
			outputs = model(images)
			loss = criterion(outputs, labels)

		loss.backward()
		optimizer.step()

		_, preds = torch.max(outputs, 1)
		running_loss += loss.item() * images.size(0)
		running_corrects += torch.sum(preds == labels).item()
		total += images.size(0)

	avg_loss = running_loss / max(1, total)
	avg_acc = running_corrects / max(1, total)
	return avg_loss, avg_acc


def evaluate(model, loader, criterion, device) -> Tuple[float, float]:

	model.eval()
	running_loss = 0.0
	running_corrects = 0
	total = 0
	with torch.no_grad():
		for images, labels in loader:
			images = images.to(device)
			labels = labels.to(device)
			outputs = model(images)
			loss = criterion(outputs, labels)
			_, preds = torch.max(outputs, 1)
			running_loss += loss.item() * images.size(0)
			running_corrects += torch.sum(preds == labels).item()
			total += images.size(0)

	avg_loss = running_loss / max(1, total)
	avg_acc = running_corrects / max(1, total)
	return avg_loss, avg_acc


def main():

	parser = argparse.ArgumentParser(description="Train skin defects classifier")
	parser.add_argument("--data_dir", default="Project102/Dataset/train", type=str, help="Path to folder-based dataset (class subfolders)")
	parser.add_argument("--csv", default=None, type=str, help="Path to CSV metadata (optional, overrides --data_dir)")
	parser.add_argument("--files_root", default="files", type=str, help="Root folder for CSV-based images")
	parser.add_argument("--epochs", default=30, type=int)
	parser.add_argument("--batch_size", default=16, type=int)
	parser.add_argument("--image_size", default=256, type=int)
	parser.add_argument("--lr", default=3e-4, type=float)
	parser.add_argument("--arch", default="convnext_tiny", type=str, choices=["efficientnet_b0", "resnet18", "convnext_tiny"], help="Backbone architecture")
	parser.add_argument("--mixup", default=0.2, type=float, help="Mixup alpha; 0 disables")
	parser.add_argument("--cutmix", default=0.0, type=float, help="CutMix alpha; 0 disables")
	parser.add_argument("--freeze_epochs", default=2, type=int, help="Freeze backbone for first N epochs")
	parser.add_argument("--warmup_epochs", default=2, type=int, help="LR warmup epochs")
	parser.add_argument("--patience", default=8, type=int, help="Early stopping patience (epochs)")
	parser.add_argument("--val_size", default=0.2, type=float)
	parser.add_argument("--seed", default=42, type=int)
	parser.add_argument("--freeze_backbone", action="store_true")
	parser.add_argument("--out_dir", default="artifacts", type=str)
	args = parser.parse_args()
	# expose args for train_one_epoch (for mixup/cutmix)
	global args_holder
	args_holder = args

	random.seed(args.seed)
	torch.manual_seed(args.seed)
	os.makedirs(args.out_dir, exist_ok=True)

	# Load dataset from CSV or folder structure
	if args.csv is not None:
		print(f"Loading dataset from CSV: {args.csv}")
		items, class_names = expand_csv_rows_to_images(args.csv, args.files_root)
	else:
		print(f"Loading dataset from folder: {args.data_dir}")
		items, class_names = load_folder_dataset(args.data_dir)
	
	if len(class_names) < 2:
		raise RuntimeError("Need at least 2 classes to train a classifier.")
	print(f"Found {len(items)} images across {len(class_names)} classes: {class_names}")
	# Print class counts
	from collections import Counter
	counts = Counter([lbl for _, lbl, _ in items])
	print(f"Class counts: {dict(counts)}")

	class_to_idx = {name: i for i, name in enumerate(class_names)}
	with open(os.path.join(args.out_dir, "class_to_idx.json"), "w", encoding="utf-8") as f:
		json.dump(class_to_idx, f, ensure_ascii=False, indent=2)

	train_loader, val_loader = build_dataloaders(
		items=items,
		class_to_idx=class_to_idx,
		image_size=args.image_size,
		batch_size=args.batch_size,
		val_size=args.val_size,
		seed=args.seed,
	)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = create_model(arch=args.arch, num_classes=len(class_names), freeze_backbone=args.freeze_backbone).to(device)
	criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
	optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)

	# Warmup + cosine schedule
	def lr_lambda(current_epoch: int):
		if current_epoch < args.warmup_epochs:
			return float(current_epoch + 1) / float(max(1, args.warmup_epochs))
		progress = (current_epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
		from math import pi, cos
		return 0.5 * (1.0 + cos(pi * min(1.0, max(0.0, progress))))

	scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

	best_val_acc = 0.0
	model_path = os.path.join(args.out_dir, "model.pth")
	no_improve = 0
	
	# Clean up old model files if they exist
	old_best_path = os.path.join(args.out_dir, "model_best.pth")
	old_last_path = os.path.join(args.out_dir, "model_last.pth")
	if os.path.exists(old_best_path):
		os.remove(old_best_path)
	if os.path.exists(old_last_path):
		os.remove(old_last_path)

	# Optionally freeze backbone for initial epochs
	def set_backbone_requires_grad(require: bool):
		for name, param in model.named_parameters():
			if "classifier" in name or name.endswith(".fc.weight") or name.endswith(".fc.bias"):
				# keep head as is
				continue
			param.requires_grad = require

	for epoch in range(1, args.epochs + 1):
		if epoch == 1 and args.freeze_epochs > 0:
			set_backbone_requires_grad(False)
			optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
		elif epoch == args.freeze_epochs + 1:
			set_backbone_requires_grad(True)
			optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
		train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
		val_loss, val_acc = evaluate(model, val_loader, criterion, device)
		print(
			f"Epoch {epoch:02d}/{args.epochs} | "
			f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
			f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
		)

		scheduler.step()

		# Save model when validation accuracy improves, delete old model first
		if val_acc >= best_val_acc:
			best_val_acc = val_acc
			# Delete old model if it exists
			if os.path.exists(model_path):
				os.remove(model_path)
			torch.save({
				"model_state_dict": model.state_dict(),
				"class_to_idx": class_to_idx,
				"image_size": args.image_size,
				"arch": args.arch,
			}, model_path)
			no_improve = 0
		else:
			no_improve += 1
			if no_improve >= args.patience:
				print("Early stopping due to no improvement")
				break

	print(f"Model saved to {model_path}")


if __name__ == "__main__":
	main()


