# Document Assistant — RAG-система 


**Проект создан в рамках технического задания для стажировки в ПАО «Северста́ль»**  
**Среда разработки**: Kaggle Notebook  

**Почему Kaggle?**  
 - Бесплатный доступ к мощным GPU для работы с LLM
 - Воспроизводимость результатов
 - Возможность чтения и записи данных
 
## Инструкция по началу работы с Kaggle
1. Установите в начале кода зависимости:

    `!pip install langchain langchain-community langchain-text-splitters faiss-cpu sentence-transformers torch transformers pypdf docx2txt`
3. Загрузите документы в датасет с именем `test_task` или укажите свои директории
4. Не забудьте включить Internet и accelerator GPU T4x2
## Примеры использования

### Сценарий 1

**Файл:** `A9RD3D4.pdf`

**Вопрос:** "Какая цель политики Северстали?"

**Ответ системы:**  "Предотвращение и минимизация потерь АО «Северсталь Менеджмент» и управляемых им обществ от утечек конфиденциальной информации, использования недостоверной, искаженной информации и нарушений процессов обработки информации."

### Сценарий 2

**Файл:** `Polzovatelskoe_soglashenie.pdf`

**Вопрос:** "Какой срок действия настоящего Соглашения?"

**Ответ системы:** " Срок действия настоящего Соглашения не ограничен."

### Сценарий 3

**Файл:** `University Success.docx`

**Вопрос:** "Who is George Draper Dayton?"

**Ответ системы:** "George Draper Dayton was the founder of Target Corporation. He started the company as Dayton Dry Goods Company in Minnesota, US, in 1902. In 1962, the company was renamed Target with the goal of offering retail goods at a good price, with good service and a commitment to the community."

 (в файле results.json сохранены все примеры запросов, ответов, а также релевантные чанки)


## О проекте

### Технологический стек

| Компонент | Технология | Назначение |
|-----------|------------|------------|
| **Обработка документов** | `langchain_community.document_loaders` | Загрузка PDF и DOCX |
| **Разделение на чанки** | `langchain_text_splitters` | Рекурсивное разделение на чанки |
| **Эмбеддинги** | `sentence-transformers` + `FAISS` | Семантическое представление и поиск |
| **Языковая модель** | `transformers` (Hugging Face) | Генерация ответов |
| **Логирование** | `logging` | Логирование процесса работы |
| **Управление памятью** | `torch.cuda`, `gc` | Оптимизация использования GPU/CPU |

### Поддерживаемые форматы документов

- **PDF** (.pdf) — через `PyPDFLoader`
- **Word** (.docx, .doc) — через `Docx2txtLoader`

### Используемые модели

- **`all-MiniLM-L6-v2`** — легкая модель от SentenceTransformers для вычисления эмбеддингов

- **`Qwen/Qwen2.5-3B`** — маленькая модель для генерации ответов

### Настройка параметров

При инициализации `DocumentAssistant` можно настроить:

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `chunk_size` | 1024 | Максимальный размер чанка в символах |
| `overlap*` | 100 | Перекрытие между соседними чанками |
| `top_k` | 3 | Количество релевантных фрагментов для поиска |
| `filepath` | `/kaggle/working/result.json` | Путь для сохранения результатов |
| `embedding_model_name` | `all-MiniLM-L6-v2` | Модель для эмбеддингов |
| `llm_model_name` | `Qwen/Qwen2.5-3B` | Языковая модель |

***Важно:** значение overlap не должно превышать chunk_size
