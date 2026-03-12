"""Image embedding and labeling utilities using CLIP via sentence-transformers.

We use a lightweight CLIP model to extract image embeddings and predict coarse labels
by comparing to a small vocabulary (COCO-80 classes). Labels are used to guide web search.
"""
from __future__ import annotations

from typing import List, Tuple
from io import BytesIO

from PIL import Image
import numpy as np

# A compact vocabulary of common objects (COCO-80 classes)
COCO80 = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]


class ImageEmbedder:
    """CLIP-based image embedder and zero-shot label predictor."""

    def __init__(self, model_name: str = "clip-ViT-B-32"):
        from sentence_transformers import SentenceTransformer
        # This model supports encode_image and encode for PIL inputs
        self.model = SentenceTransformer(model_name)
        # Pre-encode vocabulary texts once for faster label prediction
        self._label_texts = [f"a photo of a {w}" for w in COCO80]
        self._label_emb = self.model.encode(self._label_texts, convert_to_numpy=True, normalize_embeddings=True)

    def _load_image(self, image_bytes: bytes) -> Image.Image:
        return Image.open(BytesIO(image_bytes)).convert("RGB")

    def embed_image(self, image_bytes: bytes) -> List[float]:
        img = self._load_image(image_bytes)
        vec = self.model.encode([img], convert_to_numpy=True, normalize_embeddings=True)[0]
        return vec.tolist()

    def predict_labels(self, image_bytes: bytes, top_k: int = 5) -> List[str]:
        img = self._load_image(image_bytes)
        img_emb = self.model.encode([img], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = (self._label_emb @ img_emb)  # cosine since normalized
        idx = np.argsort(-sims)[:top_k]
        return [COCO80[i] for i in idx]
