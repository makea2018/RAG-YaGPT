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

# Streamlit
import streamlit as st
import json


load_dotenv()

# Загрузка переменных
CATALOG_NAME = os.getenv("CATALOG_NAME")
API_KEY = os.getenv("API_KEY")

# Cоздаем объект YandexGPTEmbeddings для построения векторов с помощью YandexGPT
embeddings = YandexEmbeddings(api_key=API_KEY, folder_id=CATALOG_NAME)

# Подключение векторной БД
db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
top_k = 2
retriever = db.as_retriever(search_kwargs={'k': top_k})

# Определение LLM
instructions = """
Представь себе, что ты сотрудник Yandex Cloud. Твоя задача - вежливо и по мере своих сил отвечать на все вопросы собеседника."""
llm = YandexLLM(api_key=API_KEY, folder_id=CATALOG_NAME,
                instruction_text=instructions, temperature=0.3,
                max_tokens=320, model=0)

# Промпт для языковой модели
document_variable_name = "context"
rag_template = """
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

# Streamlit app
st.title("RAG YaChatbot")

# Создаем экземпляр истории сообщений
message_history = StreamlitChatMessageHistory()

# Создание папки, в которой будут храниться Истории Чатов
os.makedirs("Chats_history", exist_ok=True)

# Функции для загрузки и сохранения истории сообщений
def load_message_history():
    if os.path.exists("Chats_history/message_history.json"):
        with open("Chats_history/message_history.json", "r", encoding="utf-8") as file:
            return json.load(file)
    return []

def save_message_history(messages):
    with open("Chats_history/message_history.json", "w", encoding="utf-8") as file:
        json.dump(messages, file, ensure_ascii=False, indent=4)

# Загрузка истории сообщений из файла
stored_messages = load_message_history()
message_history.messages = stored_messages

if len(message_history.messages) == 0:
    message_history.add_message({"type": "assistant", "content": "Здравствуйте! Как я могу вам помочь?"})
for msg in message_history.messages:
    st.chat_message(msg["type"]).write(msg["content"])

# Сохранение истории сообщений в файл после каждого обмена
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    # Добавляем сообщение пользователя в историю
    message_history.add_message({"type": "user", "content": prompt})
    
    # Получаем ответ ассистента
    output = rag_chain.invoke(prompt)
    
    # Добавляем ответ ассистента в историю
    st.chat_message("ai").write(output)
    message_history.add_message({"type": "assistant", "content": output})

    # Сохраняем обновленную историю сообщений в файл
    save_message_history(message_history.messages)
