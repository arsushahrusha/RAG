from typing import Callable, List, Tuple, Dict
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
        self.dense_index = None

    def load_model(self) -> None:
        from sentence_transformers import SentenceTransformer
        if self._is_loaded: return
        self._model = SentenceTransformer(self.embedding_model_name)
        self._is_loaded = True

    def get_embedding(self) -> Dict[str, np.ndarray]:
        import faiss
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
        dimension = embeddings.shape[1]
        self.dense_index = faiss.IndexFlatIP(dimension)

        faiss.normalize_L2(embeddings)
        self.dense_index.add(embeddings.astype('float32'))

        return dict(zip(self.documents, embeddings))

    def _dense_search(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        import faiss
        if not self._is_loaded: self.load_model()
        query_embedding = self._model.encode([query], convert_to_numpy = True)
        faiss.normalize_L2(query_embedding)
        scores, indices = self.dense_index.search(query_embedding.astype('float32'), top_k)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]

    def reciprocal_rank_fusion(self, dense_result: List[Tuple[int, float]], k: int = 60) -> List[Tuple[int, float]]:
        rrf_scores = {}
        
        for rank, (doc_idx, _) in enumerate(dense_result, 1):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0)+1/(k+rank)

        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        dense_result = self._dense_search(query, top_k)
        final_result = self.reciprocal_rank_fusion(dense_result)
        results = []
        for doc_idx, score in final_result[:top_k]:
            results.append({
                'document': self.documents[doc_idx],
                'score': score,
                'index': doc_idx
            })
        return results
    
chunks = doc_assist.split_text(res)
embed_model = EmbeddingModel(chunks)
embeddings = embed_model.get_embedding()

query = 'What Is a Business Model?'
results = embed_model.search(query, top_k=3)
res = []
for i, res in enumerate(results):
    res.append(res['document'])

class RAGModel:
    def __init__(self, retrieved_chunks: List[str], query: str, model_name: str = "Qwen/Qwen2.5-3B") -> None:
        self.retrieved_chunks = retrieved_chunks
        self.query = query
        self.model_name = model_name
        self._model = None
        self._is_loaded = False
        self._current_pipeline = None
        self._current_tokenizer = None
    
    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            use_cache=True,
            device_map="auto",
            trust_remote_code=True,
        )

        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=8192,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.2,
            return_full_text=True
        )

        self._current_model = model
        self._current_pipeline = hf_pipeline
        self._current_tokenizer = tokenizer
        self._is_loaded = True
        return hf_pipeline
    
    def create_prompt(self) -> str:
        prompt = f'''Используй только следующие фрагменты документов для ответа: {self.retrieved_chunks}
Вопрос: {self.query}
Ответ:
'''
        return prompt
    
    def get_answer(self):
        if not self._is_loaded: self.load_model()
        llm = self._current_model
        tokenizer = self._current_tokenizer

        prompt = self.create_prompt()

        inputs = tokenizer(
                prompt,  
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=2048 
            ).to("cuda")
        
        try:
            with torch.no_grad():
                outputs = llm.generate(
                    **inputs,
                    max_new_tokens=8192, 
                    temperature=0.2,
                    do_sample=True,     
                    num_beams=1,         
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,   
                    repetition_penalty=1.0,  
                    output_scores=False,  
                    return_dict_in_generate=True, 
                )
            result = tokenizer.decode(outputs.sequences[0], skip_special_tokens = True)
            return result
        except Exception as e:
            print(f"Ошибка при обработке вопроса: {e}")

rag = RAGModel(res, query)
print(rag.get_answer())
            

