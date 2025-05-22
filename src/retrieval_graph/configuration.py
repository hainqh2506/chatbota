import tiktoken
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import AutoTokenizer
from langchain_together import ChatTogether , Together
from dotenv import load_dotenv
import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from typing import List, Optional, Any
from langchain_core.embeddings import Embeddings
import numpy as np
# Tạo wrapper class cho SentenceTransformer
EMBEDDING_MODEL = "AITeamVN/Vietnamese_Embedding"
load_dotenv()
class VietnameseEmbeddings(Embeddings):
    """Singleton Embeddings for Vietnamese using SentenceTransformer."""
    _instance: Optional['VietnameseEmbeddings'] = None

    def __new__(cls, model_name: str = EMBEDDING_MODEL):
        # Nếu chưa có instance, tạo mới
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Khởi tạo model chỉ một lần
            cls._instance._initialize_model(model_name)
        return cls._instance

    def _initialize_model(self, model_name: str):
        try:
            print(f"Initializing Vietnamese embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Error initializing embedding model: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()
    def embed_text(self, text: str)-> np.ndarray:  #using in RAPTOR (np.ndarray)
        return self.model.encode(text, convert_to_numpy=True)

# Hàm tải tokenizer
def load_tokenizer(tokenizer_model: str = EMBEDDING_MODEL):
    """
    Hàm tải và khởi tạo tokenizer.
    
    Tham số:
    - tokenizer_model: Tên mô hình tokenizer. Mặc định là "keepitreal/vietnamese-sbert".
    
    Trả về: Đối tượng tokenizer.
    """
    return AutoTokenizer.from_pretrained(tokenizer_model)

# Hàm tải mô hình embedding

# def load_embedding_model_VN(embedding_model: str = "dangvantuan/vietnamese-embedding"):
#     """
#     Hàm tải và khởi tạo mô hình embedding.
    
#     Tham số:
#     - embedding_model: Tên mô hình embedding.
    
#     Trả về: Đối tượng VietnameseEmbeddings.
#     """
#     return VietnameseEmbeddings(embedding_model)
def load_embedding_VN(embedding_model: str = "AITeamVN/Vietnamese_Embedding"):
    return VietnameseEmbeddings(embedding_model)
def load_embedding_VN2(embedding_model: str = "huyydangg/DEk21_hcmute_embedding"):
    return VietnameseEmbeddings(embedding_model)
# Hàm tải mô hình embedding
def load_embedding(embedding_model: str = EMBEDDING_MODEL):

    return HuggingFaceEmbeddings(model_name=embedding_model)

# gemini
def load_gemini(model_name: str = "gemini-2.0-flash-exp"): #-8b
        # Load API keys từ .env file
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    llm = ChatGoogleGenerativeAI(
    model=model_name,
    temperature=0.5,
    #max_tokens=512,
    api_key = GEMINI_API_KEY
    # other params...
)
    return llm

def load_together_model(model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"):  
    load_dotenv()
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
    return ChatTogether(api_key=TOGETHER_API_KEY, model_name=model_name)
    
# Hàm tải và khởi tạo tokenizer từ tiktoken (nếu cần)
def load_tiktoken(tokenizer_name: str = "o200k_base"):  #cl100k_base
    """
    Hàm tải và khởi tạo tokenizer từ tiktoken.
    
    Tham số:
    - tokenizer_name: Tên tokenizer của tiktoken. Mặc định là "cl100k_base".
    
    Trả về: Đối tượng tokenizer.
    """
    return tiktoken.get_encoding(tokenizer_name)
