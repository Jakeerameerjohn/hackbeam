import os
from dotenv import load_dotenv
load_dotenv()

KB_FILE_PATH = "./data"
COLLECTION_NAME = "lightcast_kb_collection"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
HF_LLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_LLM_MODEL_ID}"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
