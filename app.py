from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough

# Для RAG системы
from langchain_chroma import Chroma
from yandex_chain.YandexGPT import YandexLLM
from yandex_chain.YandexGPTEmbeddings import YandexEmbeddings
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import os

load_dotenv()

# Загрузка переменных
CATALOG_NAME = os.getenv("CATALOG_NAME")
API_KEY = os.getenv("API_KEY")

# Cоздаем объект YandexGPTEmbeddings для построения векторов с помощью YandexGPT
embeddings = YandexEmbeddings(api_key=API_KEY, folder_id=CATALOG_NAME)

# Подключение векторной БД
db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
top_k = 3
retriever = db.as_retriever(search_kwargs={'k': top_k})

# Определение LLM
instructions = """
Представь себе, что ты сотрудник Yandex Cloud. Твоя задача - вежливо и по мере своих сил отвечать на все вопросы собеседника."""
llm = YandexLLM(api_key=API_KEY, folder_id=CATALOG_NAME,
                instruction_text=instructions)

# Промпт для языковой модели
document_variable_name = "context"
rag_template = """
    Пожалуйста, посмотри на текст ниже и ответь на вопрос, используя информацию из этого текста.
    Текст:
    -----
    {context}
    -----
    Вопрос:
    {query}
"""

# Определяем rag-prompt
rag_prompt = PromptTemplate.from_template(rag_template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Создаём цепочку
rag_chain = (
    {"context": retriever | format_docs, "query": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Сборка всех компонентов в единную систему
print(rag_chain.invoke("Дай определение рудовоза"))
