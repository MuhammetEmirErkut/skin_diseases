import argparse
import json
import os
from typing import Dict

from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms


def load_checkpoint(checkpoint_path: str):

	ckpt = torch.load(checkpoint_path, map_location="cpu")
	class_to_idx: Dict[str, int] = ckpt["class_to_idx"]
	image_size: int = ckpt.get("image_size", 224)
	arch: str = ckpt.get("arch", "resnet18")

	idx_to_class = {v: k for k, v in class_to_idx.items()}

	arch = arch.lower()
	if arch == "resnet18":
		model = models.resnet18(weights=None)
		in_features = model.fc.in_features
		model.fc = nn.Linear(in_features, len(class_to_idx))
	elif arch == "efficientnet_b0":
		model = models.efficientnet_b0(weights=None)
		in_features = model.classifier[-1].in_features
		model.classifier[-1] = nn.Linear(in_features, len(class_to_idx))
	elif arch == "convnext_tiny":
		model = models.convnext_tiny(weights=None)
		in_features = model.classifier[-1].in_features
		model.classifier[-1] = nn.Linear(in_features, len(class_to_idx))
	else:
		raise ValueError(f"Unknown arch in checkpoint: {arch}")
	model.load_state_dict(ckpt["model_state_dict"])
	model.eval()

	preprocess = transforms.Compose([
		transforms.Resize((image_size, image_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	return model, preprocess, idx_to_class


def predict(model, preprocess, idx_to_class, image_path: str, device: torch.device, tta: int = 1):

	image = Image.open(image_path).convert("RGB")
	base = preprocess(image).unsqueeze(0).to(device)
	tensors = [base]
	if tta > 1:
		hflip = transforms.functional.hflip(image)
		tensors.append(preprocess(hflip).unsqueeze(0).to(device))
	with torch.no_grad():
		outputs = []
		model = model.to(device)
		for t in tensors:
			outputs.append(model(t))
		outputs = torch.stack(outputs, dim=0).mean(dim=0)
		probs = torch.softmax(outputs, dim=1)[0]
		conf, pred_idx = torch.max(probs, dim=0)
		pred_class = idx_to_class[pred_idx.item()]
		return pred_class, conf.item(), probs.cpu().tolist()


def main():

	parser = argparse.ArgumentParser(description="Run inference on a single image")
	parser.add_argument("--checkpoint", default=os.path.join("artifacts", "model.pth"), type=str)
	parser.add_argument("--image", required=True, type=str)
	parser.add_argument("--tta", default=2, type=int, help="Test-time augmentation passes (1-2)")
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model, preprocess, idx_to_class = load_checkpoint(args.checkpoint)
	pred_label, confidence, probs = predict(model, preprocess, idx_to_class, args.image, device, tta=args.tta)

	print(json.dumps({
		"predicted": pred_label,
		"confidence": round(confidence, 4),
		"probs": {idx_to_class[i]: round(p, 4) for i, p in enumerate(probs)},
	}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()


