from typing import Callable, List
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os       
import numpy as np

pdf_file = '/kaggle/input/test-task/A9RD3D4.pdf'
dox_file = '/kaggle/input/test-task/University Success.docx'
file_paths = ['/kaggle/input/test-task/A9RD3D4.pdf', '/kaggle/input/test-task/University Success.docx']


class Downloader:
    def __init__(self, file_paths: List[str]) -> None:
        self.file_paths = file_paths

    def get_file_extension(self, filename: str) -> str:
        ext = os.path.splitext(filename)[1].lower()
        if ext == '.pdf': return 'PDF'
        elif ext in ['.docx', '.doc']: return 'DOCX'
        else: return 'UNKNOWN'
    
    def get_file_text(self) -> List[List[str]]:
        result = []
        for filename in self.file_paths:
            if self.get_file_extension(filename) == 'PDF':
                loader = PyPDFLoader(filename)
                documents = loader.load()
                result.append([doc.page_content for doc in documents])
            elif self.get_file_extension(filename) == 'DOCX':
                text_loader = Docx2txtLoader(filename)
                documents = text_loader.load()
                result.append([doc.page_content for doc in documents])
            else: 
                return 'Невозможно обработать файл: неподдерживаемый тип данных'
        return result


dwn = Downloader(file_paths)
res = dwn.get_file_text()

class DocumentAssistant:
    def __init__(self, chunk_size: int, chunk_overlap: int, separators: List[str] = None, length_function: Callable = len) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ".", " ", ""]
        self.length_function = length_function
    
    def split_text(self, texts: List[List[str]]) -> List[str]:
        all_pages = []
        for pages in texts:
            all_pages.extend(pages)


        full_text = '\n'.join(all_pages)
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            length_function = len,
            separators = self.separators
        )

        split_documents = splitter.split_text(full_text)
        return(split_documents)
    
doc_assist = DocumentAssistant(1000, 100)

class EmbeddingModel:
    def __init__(self, documents: List[str], embedding_model_name: str = "all-MiniLM-L6-v2") -> None:
        self.documents = documents
        self.embedding_model_name = embedding_model_name
        self._model = None
        self._is_loaded = False
        self.chunk_metadata = []

    def load_model(self) -> None:
        from sentence_transformers import SentenceTransformer
        if self._is_loaded: return
        self._model = SentenceTransformer(self.embedding_model_name)
        self._is_loaded = True

    def get_embedding(self) -> np.ndarray:
        if not self._is_loaded: self.load_model()
        embeddings = self._model.encode(
            self.documents,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings = True
        )
        print(f'Размер: {embeddings[0].shape}')
        print(f'Количество эмбеддингов: {len(embeddings)}')
        return embeddings

chunks = doc_assist.split_text(res)
embed_model = EmbeddingModel(chunks)
embeddings = embed_model.get_embedding()

for i in embeddings:
    print(i)





