import os
import json
import io
from typing import Dict, Tuple, List

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

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


def predict(model, preprocess, idx_to_class, pil_image: Image.Image, device: torch.device):

	tensor = preprocess(pil_image.convert("RGB")).unsqueeze(0).to(device)
	with torch.no_grad():
		outputs = model.to(device)(tensor)
		probs = torch.softmax(outputs, dim=1)[0]
		conf, pred_idx = torch.max(probs, dim=0)
		pred_class = idx_to_class[pred_idx.item()]
		prob_map = {idx_to_class[i]: float(probs[i].item()) for i in range(len(idx_to_class))}
		return pred_class, float(conf.item()), prob_map


class SkinGUI:

	def __init__(self, root: tk.Tk) -> None:
		self.root = root
		self.root.title("Cilt Hastalığı Sınıflandırma")
		self.root.geometry("900x600")

		self.model = None
		self.preprocess = None
		self.idx_to_class = None
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.image_paths: List[str] = []
		self.tk_image = None

		# Top controls
		ctrl_frame = tk.Frame(self.root)
		ctrl_frame.pack(fill=tk.X, padx=10, pady=10)

		self.ckpt_var = tk.StringVar(value=os.path.join("artifacts", "model.pth"))
		tk.Label(ctrl_frame, text="Model dosyası:").pack(side=tk.LEFT)
		self.ckpt_entry = tk.Entry(ctrl_frame, textvariable=self.ckpt_var, width=60)
		self.ckpt_entry.pack(side=tk.LEFT, padx=5)
		tk.Button(ctrl_frame, text="Seç", command=self.choose_ckpt).pack(side=tk.LEFT)
		tk.Button(ctrl_frame, text="Modeli Yükle", command=self.load_model).pack(side=tk.LEFT, padx=5)

		# Left: listbox for files
		left_frame = tk.Frame(self.root)
		left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

		tk.Button(left_frame, text="Resim(ler) Seç", command=self.choose_images).pack(fill=tk.X)
		self.listbox = tk.Listbox(left_frame, width=40, height=25)
		self.listbox.pack(fill=tk.BOTH, expand=True, pady=8)
		self.listbox.bind("<<ListboxSelect>>", self.on_select)

		# Right: image + prediction
		right_frame = tk.Frame(self.root)
		right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

		self.image_label = tk.Label(right_frame, text="Görsel önizleme")
		self.image_label.pack(fill=tk.BOTH, expand=True)

		self.pred_label_var = tk.StringVar(value="Tahmin: -")
		self.pred_label = tk.Label(right_frame, textvariable=self.pred_label_var, font=("Arial", 14))
		self.pred_label.pack(anchor=tk.W, pady=8)

		self.prob_text = tk.Text(right_frame, height=14)
		self.prob_text.pack(fill=tk.BOTH, expand=True)

	def choose_ckpt(self):
		path = filedialog.askopenfilename(filetypes=[("PyTorch Checkpoint", "*.pth")])
		if path:
			self.ckpt_var.set(path)

	def load_model(self):
		ckpt_path = self.ckpt_var.get().strip()
		if not os.path.isfile(ckpt_path):
			messagebox.showerror("Hata", f"Model dosyası bulunamadı: {ckpt_path}")
			return
		try:
			self.model, self.preprocess, self.idx_to_class = load_checkpoint(ckpt_path)
			messagebox.showinfo("Bilgi", "Model yüklendi.")
		except Exception as e:
			messagebox.showerror("Hata", f"Model yüklenemedi: {e}")

	def choose_images(self):
		paths = filedialog.askopenfilenames(filetypes=[("Images", "*.jpg *.jpeg *.png")])
		if not paths:
			return
		self.image_paths = list(paths)
		self.listbox.delete(0, tk.END)
		for p in self.image_paths:
			self.listbox.insert(tk.END, os.path.basename(p))

	def on_select(self, event):
		if not self.listbox.curselection():
			return
		idx = self.listbox.curselection()[0]
		path = self.image_paths[idx]
		self.show_image_and_predict(path)

	def show_image_and_predict(self, path: str):
		try:
			pil_img = Image.open(path).convert("RGB")
		except Exception as e:
			messagebox.showerror("Hata", f"Görsel açılamadı: {e}")
			return

		# Resize preview keeping aspect ratio
		preview = pil_img.copy()
		preview.thumbnail((600, 400))
		self.tk_image = ImageTk.PhotoImage(preview)
		self.image_label.configure(image=self.tk_image)

		if self.model is None:
			self.pred_label_var.set("Tahmin: (Önce modeli yükleyin)")
			self.prob_text.delete(1.0, tk.END)
			return

		pred, conf, probs = predict(self.model, self.preprocess, self.idx_to_class, pil_img, self.device)
		self.pred_label_var.set(f"Tahmin: {pred} ({conf:.2%})")
		self.prob_text.delete(1.0, tk.END)
		for k, v in sorted(probs.items(), key=lambda x: x[1], reverse=True):
			self.prob_text.insert(tk.END, f"{k}: {v:.4f}\n")


def main():

	root = tk.Tk()
	app = SkinGUI(root)
	root.mainloop()


if __name__ == "__main__":
	main()


