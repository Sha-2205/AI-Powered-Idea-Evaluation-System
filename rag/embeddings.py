import os
from typing import List, Union
import numpy as np


os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model_cache = {}


def get_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """
    Loads and caches the SentenceTransformer model.

    Args:
        model_name (str): Hugging Face model identifier.

    Returns:
        SentenceTransformer: Loaded model instance.
    """
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def get_embeddings(
    texts: Union[str, List[str]],
    model_name: str = DEFAULT_MODEL_NAME
) -> np.ndarray:
    """
    Converts a single text string or list of text strings into normalized numerical embeddings.

    Args:
        texts (str | List[str]): Text or list of texts to embed.
        model_name (str): Embedding model to use.

    Returns:
        np.ndarray: 2D numpy array of shape (N, dimension) with float32 dtype.
    """
    model = get_embedding_model(model_name)

    if isinstance(texts, str):
        texts = [texts]

    
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return embeddings.astype(np.float32)
