import io
import json
import os
from typing import Dict, Tuple

import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms


@st.cache_resource(show_spinner=False)
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


def predict(model, preprocess, idx_to_class, pil_image: Image.Image, device: torch.device) -> Tuple[str, float, Dict[str, float]]:

	tensor = preprocess(pil_image.convert("RGB")).unsqueeze(0).to(device)
	with torch.no_grad():
		outputs = model.to(device)(tensor)
		probs = torch.softmax(outputs, dim=1)[0]
		conf, pred_idx = torch.max(probs, dim=0)
		pred_class = idx_to_class[pred_idx.item()]
		prob_map = {idx_to_class[i]: float(probs[i].item()) for i in range(len(idx_to_class))}
		return pred_class, float(conf.item()), prob_map


def main():

	st.set_page_config(page_title="Skin Disease Classifier", page_icon="🩺", layout="centered")
	st.title("🩺 Cilt Hastalığı Sınıflandırma")
	st.caption("Resim yükleyin, model hangi sınıfa ait olduğunu tahmin etsin.")

	with st.sidebar:
		st.header("Ayarlar")
		checkpoint = st.text_input("Model dosyası", value=os.path.join("artifacts", "model.pth"))
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		st.write(f"Cihaz: {device}")

	# Load model lazily
	load_btn = st.button("Modeli Yükle")
	model = preprocess = idx_to_class = None
	if load_btn or os.path.exists(os.path.join("artifacts", "model.pth")):
		try:
			model, preprocess, idx_to_class = load_checkpoint(checkpoint)
			st.success("Model yüklendi.")
		except Exception as e:
			st.error(f"Model yüklenemedi: {e}")

	uploaded_files = st.file_uploader("Resim(ler)i yükleyin", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
	if uploaded_files and model is not None:
		cols = st.columns(1)
		for up in uploaded_files:
			try:
				image = Image.open(io.BytesIO(up.read()))
			except Exception:
				st.warning(f"{up.name} okunamadı.")
				continue

			pred_label, confidence, probs = predict(model, preprocess, idx_to_class, image, device)

			st.subheader(up.name)
			st.image(image, caption=f"Tahmin: {pred_label} ({confidence:.2%})", width='stretch')
			with st.expander("Olasılıklar"):
				st.json({k: round(v, 4) for k, v in sorted(probs.items(), key=lambda x: x[1], reverse=True)})

	elif uploaded_files and model is None:
		st.info("Önce modeli yükleyin.")

	st.markdown("---")
	st.caption("Model: ResNet18 fine-tune. Daha fazla sınıf eklemek için CSV ve veri klasörlerini genişletin.")


if __name__ == "__main__":
	main()


