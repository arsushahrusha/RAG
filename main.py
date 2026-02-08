from typing import Callable, List, Tuple, Dict
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os  
import gc
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import json
import logging
import sys
import datetime

def setup_logging():
    logging.basicConfig(
        level = logging.INFO,
        format='%(asctime)s-%(name)s-%(levelname)s-%(message)s',
        datefmt='%Y-%m-%d   %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

setup_logging()
logger = logging.getLogger(__name__)

class Downloader:
    def __init__(self, file_paths: List[str]) -> None:
        self.file_paths = file_paths

    def get_file_extension(self, filename: str) -> str:
        ext = os.path.splitext(filename)[1].lower()
        logger.info('Получаем расширение...')
        if ext == '.pdf': return 'PDF'
        elif ext in ['.docx', '.doc']: return 'DOCX'
        else: return 'UNKNOWN'
        
    
    def get_file_text(self) -> List[List[str]]:
        result = []
        for filename in self.file_paths:
            try:
                if not os.path.exists(filename):
                    logger.error(f'Файл не найден: {filename}')
                    return f'ОШИБКА: Файл не найден: {filename}'
                ext = self.get_file_extension(filename)
                if ext == 'PDF':
                    try:
                        loader = PyPDFLoader(filename)
                        documents = loader.load()
                        result.append([doc.page_content for doc in documents])
                    except Exception as e:
                        logger.error(f'Ошибка при чтении PDF файла {filename}: {e}')
                        return f'ОШИБКА: не удалось прочитать {filename}: {e}'
                elif ext == 'DOCX':
                    try:
                        text_loader = Docx2txtLoader(filename)
                        documents = text_loader.load()
                        result.append([doc.page_content for doc in documents])
                    except Exception as e:
                        logger.error(f'Ошибка при чтении DOCX файла {filename}: {e}')
                        return f'ОШИБКА: не удалось прочитать {filename}: {e}'
                else: 
                    logger.warning(f'Неподдерживаемый формат файла: {filename}, расширение: {ext}')
                    return f'Невозможно обработать файл {filename}: неподдерживаемый тип данных'
            except Exception as e:
                logger.error(f"Ошибка при чтении файла {filename}: {e}")
                return f'ОШИБКА: не удалось прочитать {filename}: {e}'
        logger.info('Получаем текст из файлов...')
        return result

class TextSplitter:
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
        logger.info('Разделяем на чанки...')
        return split_documents

class EmbeddingModel:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2") -> None:
        self.documents = None
        self.embedding_model_name = embedding_model_name
        self._model = None
        self._is_loaded = False
        self.dense_index = None
        
    def load_model(self) -> None:
        try:
            logger.info('Загружаем модель для ембеддингов...')
            from sentence_transformers import SentenceTransformer
            if self._is_loaded: return
            self._model = SentenceTransformer(self.embedding_model_name)
            self._is_loaded = True
        except Exception as e:
            logger.error(f'Ошибка при загрузке модели: {e}')
            raise RuntimeError(f'Не удалось загрузить модель: {e}')

    def set_documents(self, documents: List[str]) -> None:
        if not documents:
            raise ValueError('Список документов не может быть пустым')
        self.documents = documents

    def get_embedding(self):
        try: 
            import faiss
            if not self._is_loaded: self.load_model()
            if self.documents is None: 
                raise ValueError("Сначала надо установить документы с помощью ф-ии set_documents()")
            embeddings = self._model.encode(
                self.documents,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings = True
            )
            
            dimension = embeddings.shape[1]
            self.dense_index = faiss.IndexFlatIP(dimension)

            faiss.normalize_L2(embeddings)
            self.dense_index.add(embeddings.astype('float32'))
            logger.info('Получаем эмбеддинги...')
        except ValueError as e:
            logger.error(f'Ошибка: неверное значение: {e}')
            raise
        except RuntimeError as e:
            logger.error(f'Ошибка выполнения: {e}')
            raise
        except Exception as e:
            logger.error(f'Ошибка при получении ембеддингов: {e}')
            raise RuntimeError(f'Ошибка при получении ембеддингов: {e}')

    def dense_search(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        try:
            import faiss
            if not self._is_loaded: self.load_model()
            if self.dense_index is None: 
                raise ValueError("Сначала надо установит индекс с помощью ф-ии get_embedding()")
            if not query: 
                raise ValueError('Запрос не может быть пустым')
            query_embedding = self._model.encode([query], convert_to_numpy = True)
            faiss.normalize_L2(query_embedding)
            scores, indices = self.dense_index.search(query_embedding.astype('float32'), top_k)
            return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]
        except ValueError as e:
            logger.error(f'Ошибка в запросе: {e}')
            raise
        except Exception as e:
            logger.error(f'Ошибка поиска: {e}')
            return []

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        try: 
            max_top_k= len(self.documents)
            if top_k > max_top_k:
                print(f"Используется top_k={max_top_k}, так как введенное значение превышает кол-во чанков")
                top_k = max_top_k
            if max_top_k == 0:
                print('Нет доступных чанков для поиска')
                return []   
            dense_result = self.dense_search(query, top_k)
            results = []
            for idx, (doc_idx, score) in enumerate(dense_result):
                results.append({
                    'document': self.documents[doc_idx],
                    'score': score,
                    'index': doc_idx,
                    'rank': idx + 1
                })
            logger.info('Ищем похожие фрагменты...')
            return results
        except ValueError as e:
            logger.error(f"Ошибка в параметрах поиска: {e}")
            raise
        except Exception as e:
            logger.error(f"Ошибка при поиске: {e}")
            return []

    def clear_memory(self):
        self.dense_index = None
        if self._model is not None: 
            del self._model
            self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        else:
            logger.warning("CUDA недоступна, невозможно очистить кеш")


class RAGModel:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B") -> None:
        self.model_name = model_name
        self._model = None
        self._is_loaded = False
        self._current_tokenizer = None
    
    def load_model(self) -> None:
        try:
            if self._is_loaded: return
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                use_cache=True,
                device_map="auto",
                trust_remote_code=True,
            )

            self._model.eval()

            self._current_tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )

            if self._current_tokenizer.pad_token is None:
                self._current_tokenizer.pad_token = self._current_tokenizer.eos_token

            self._is_loaded = True
            logger.info('Загружаем модель ЛЛМ...')
        except Exception as e:
            logger.error(f'Ошибка загрузки ЛЛМ модели: {e}')
            raise RuntimeError(f'Не удалось загрузить модель: {e}')
    
    def create_prompt(self, retrieved_chunks: List[str], query: str) -> str:
        chunks_text = '\n\n'.join([
            f'Фрагмент №{i+1}:\n{chunk}'
            for i, chunk in enumerate(retrieved_chunks)
            ])
        prompt = f'''Используй только следующие фрагменты документов для ответа: {chunks_text}
Вопрос: {query}
Ответ:
'''
        return prompt
    
    def get_answer(self, retrieved_chunks: List[str], query: str) -> str:
        if not self._is_loaded: self.load_model()
        llm = self._model
        tokenizer = self._current_tokenizer

        prompt = self.create_prompt(retrieved_chunks, query)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        inputs = tokenizer(
                prompt,  
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=4096 
            ).to(device)
        
        try:
            with torch.no_grad():
                outputs = llm.generate(
                    **inputs,
                    max_new_tokens=2048, 
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
            # Если извлекать только сгенерированный ответ
            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs.sequences[0, input_length:]   
            result = tokenizer.decode(generated_tokens, skip_special_tokens = True)
            logger.info('Получаем ответ модели...')
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при обработке вопроса: {e}")
            return f'Ошибка при генерации ответа: {str(e)}'
    
    def clear_memory(self):
        if self._model is not None:
            del self._model
            self._model = None
        if self._current_tokenizer is not None:
            del self._current_tokenizer
            self._current_tokenizer = None
        
        self._is_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        else:
            logger.warning("CUDA недоступна, невозможно очистить кеш")

class DocumentAssistant:
    def __init__(self,
                chunk_size: int = 1024,
                overlap: int = 100,
                top_k: int = 3,
                filepath: str = '/kaggle/working/result.json',
                embedding_model_name: str = 'all-MiniLM-L6-v2',
                llm_model_name: str = 'Qwen/Qwen2.5-3B') -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.filepath = filepath
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name

        self.downloader = None
        self.splitter = None
        self.chunks = None
        self.embedder = EmbeddingModel(embedding_model_name)
        self.llm = RAGModel(llm_model_name)
        open(filepath, 'w').close()

    def index_documents(self, documents: List[str]):
        logger.info('Начинаем индексирование документов...')

        # Извлекает текст из файлов
        self.downloader = Downloader(documents)
        res = self.downloader.get_file_text()
        if isinstance(res, str):
            raise ValueError(f'Ошибка загрузки файлов: {res}')
        if not res or not isinstance(res, list):
            raise ValueError('Неверный результат загрузки файлов')
        if not any(res):
            raise ValueError('Все документы пусты или не содержат текста')
    
        # Рекурсивно разбивает документы на чанки (Сначала абзацы, затем предложения и тд)
        # Сохраняет чанки
        self.splitter = TextSplitter(self.chunk_size, self.overlap)
        self.chunks = self.splitter.split_text(res)
        
        if not self.chunks:
            raise ValueError('Не получилось разбить документы на чанки, тк результат пустой')

        # Вычисляет эмбеддинги для каждого чанка с использованием sentence-transformers
        # сохраняет faiss индекс в self.embedder.dense_index
        self.embedder.set_documents(self.chunks)
        self.embedder.get_embedding()

        self.downloader = None
        self.splitter = None
        gc.collect()
    
    def answer_query(self, query: str) -> str:
        logger.info('Начинаем обрабатывать вопрос...')

        # Преобразует запрос в эмбеддинг
        # находит топ‑K наиболее близких чанков (например, K=3) по косинусному сходству
        results = self.embedder.search(query, self.top_k)
        relevant_chunks = [res['document'] for res in results]
    
        # Формирует промпт для LLM вида
        # получает ответ от LLM и возвращает его как строку
        answer = self.llm.get_answer(relevant_chunks, query)

        result_data = {
            'data_of_query': datetime.datetime.now().isoformat(),
            'query': query,
            'answer': answer,
            'relevant_chunks': relevant_chunks
        }

        with open(self.filepath, 'a', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent = 2)
        return answer
    
    def clear_all_memory(self):
        self.embedder.clear_memory()
        self.llm.clear_memory()
        
        self.chunks = None
        self.documents = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        else:
            logger.warning("CUDA недоступна, невозможно очистить кеш")

def main():
    file1 = '/kaggle/input/test-task/A9RD3D4.pdf'
    file2 = '/kaggle/input/test-task/University Success.docx'
    file3 = '/kaggle/input/test-task/Polzovatelskoe_soglashenie.pdf'
    file_paths = [file1, file2, file3]
     
    print('Создание нового DocumentAssistant...')
   
    while True:
        try:
            chunk_size = int(input("Введите максимально допустимый размер чанков: "))
            if chunk_size <= 0:
                print("ОШИБКА: размер чанков должен быть положительным числом")
                continue
            break
        except ValueError:
            print("ОШИБКА: введите целое число")

    while True:
        try:
            overlap = int(input("Введите значение для пересечения чанков: "))
            if overlap < 0:
                print("ОШИБКА: пересечение чанков не может быть отрицательным")
                continue
            if overlap >= chunk_size:
                print(f"ОШИБКА: пересечение чанков ({overlap}) должно быть меньше размера чанков ({chunk_size})")
                continue
            break
        except ValueError:
            print("ОШИБКА: введите целое число")
    
    while True:
        try:
            top_k = int(input("Введите желаемое количество релевантных фрагментов: "))
            if top_k <= 0:
                print("ОШИБКА: количество фрагментов должно быть положительным числом")
                continue
            break
        except ValueError:
            print("ОШИБКА: введите целое число")
    
    doc_assist = DocumentAssistant(chunk_size, overlap, top_k, '/kaggle/working/result.json')
    doc_assist.index_documents(file_paths)

    while True:
        
        print('''
1. Задать вопрос
2. Выйти
''')
        try:
            ans = input("Ваш выбор: ").strip()
            
            if ans == "1":
                query = input("\nВведите ваш вопрос: ").strip()
                if query:
                    answer = doc_assist.answer_query(query)
                    print(f"\nОтвет: {answer}\n")
                else:
                    print("Вопрос не может быть пустым.")
                    
            elif ans == "2":
                print("Завершение работы...")
                doc_assist.clear_all_memory()
                break
            else:
                print("ОШИБКА: выберите 1 или 2")
                
        except KeyboardInterrupt:
            print("\n\nЗавершение работы...")
            doc_assist.clear_all_memory()
            break
        except Exception as e:
            print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    main()