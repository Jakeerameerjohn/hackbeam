from sentence_transformers import SentenceTransformer
from chromadb import EmbeddingFunction, Documents, Embeddings
from backend.config import EMBEDDING_MODEL_NAME

class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(input).tolist()
