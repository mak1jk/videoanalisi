from sentence_transformers import SentenceTransformer, util
from PIL import Image
import torch

class EmbeddingsModel:
    def __init__(self, model_name="clip-ViT-B-32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP model '{model_name}' on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode_text(self, text: str):
        return self.model.encode(text, convert_to_tensor=True)

    def encode_image(self, image_path: str):
        image = Image.open(image_path)
        return self.model.encode(image, convert_to_tensor=True)
        
    def encode_images(self, image_paths: list):
        images = [Image.open(p) for p in image_paths]
        # Batch encoding
        return self.model.encode(images, batch_size=32, convert_to_tensor=True)

    def compute_similarity(self, emb1, emb2):
        return util.cos_sim(emb1, emb2)
