import os
import re
import warnings
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
import google.generativeai as genai

# Suppress future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("❌ GEMINI_API_KEY not found in environment. Please check your .env file.")
    st.stop()

# Configure Gemini model
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")
chat = model.start_chat()

# SentenceTransformer embedding wrapper for LangChain
class EmbeddingModel(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)

    def embed_query(self, text):
        return self.model.encode(text, convert_to_numpy=True)

# Load embedding model only once
@st.cache_resource
def load_embedding_model():
    return EmbeddingModel(SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2"))

wrapped_model = load_embedding_model()

# Clean WhatsApp chat messages
def clean_chat(file_path):
    pattern = re.compile(r"^\[\d{1,2}/\d{1,2}/\d{4}, \d{1,2}:\d{2}:\d{2}.*?\] ")
    messages = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if pattern.match(line):
                try:
                    line_clean = re.sub(r"^\[.*?\] ", "", line).strip()
                    if ':' in line_clean:
                        sender, message = line_clean.split(":", 1)
                        messages.append({
                            "sender": sender.strip(),
                            "text": message.strip()
                        })
                except ValueError:
                    continue
    return messages

def data_make(file_path):
    messages = clean_chat(file_path)
    sonu_lines = [msg["text"] for msg in messages if "sonu" in msg["sender"].lower()]
    documents = [Document(page_content=text) for text in sonu_lines]
    return documents

# Create or load FAISS DB
@st.cache_resource(show_spinner="Loading chat memory...")
def load_faiss_index():
    if not os.path.exists("chat.txt"):
        st.error("❌ Required file 'chat.txt' is missing.")
        st.stop()

    documents = data_make("chat.txt")

    if os.path.exists("faiss_index_chat"):
        loaded_faiss = FAISS.load_local("faiss_index_chat", embeddings=wrapped_model, allow_dangerous_deserialization=True)
    else:
        faiss_index_chat = FAISS.from_documents(documents, embedding=wrapped_model)
        faiss_index_chat.save_local("faiss_index_chat")
        loaded_faiss = faiss_index_chat

    return loaded_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 6})

##############################################################################################
# Streamlit UI and chat logic
##############################################################################################

# CSS for chat bubbles
st.markdown("""
    <style>
    .chat-container {
        display: flex;
        flex-direction: column;
    }
    .chat-message {
        display: flex;
        margin-bottom: 10px;
    }
    .message-content {
        max-width: 70%;
        padding: 10px 15px;
        border-radius: 15px;
        font-size: 16px;
        line-height: 1.4;
        word-wrap: break-word;
        position: relative;
    }
    .user {
        justify-content: flex-end;
    }
    .user .message-content {
        background-color: #dcf8c6;
        border-bottom-right-radius: 0px;
        color: black;
    }
    .sonu {
        justify-content: flex-start;
    }
    .sonu .message-content {
        background-color: #ffffff;
        border: 1px solid #ccc;
        border-bottom-left-radius: 0px;
        color: black;
    }
    .timestamp {
        font-size: 10px;
        color: #999;
        margin-top: 2px;
        text-align: right;
    }
    </style>
""", unsafe_allow_html=True)

# Chat session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load retriever (cached)
retriever = load_faiss_index()

def knowledge_base(query):
    results = retriever.invoke(query)
    return results





# Chat input


query = st.chat_input("Your message")

if query:
    now = datetime.now().strftime("%I:%M %p")

    # 1. Add user message to session
    st.session_state.messages.append({
        "role": "user",
        "content": query,
        "timestamp": now
    })

    # 2. Add "typing..." message
    st.session_state.messages.append({
        "role": "sonu",
        "content": "⏳ typing...",
        "timestamp": datetime.now().strftime("%I:%M %p")
    })

    # 3. Display current chat (user + typing)
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for entry in st.session_state.messages:
        role = entry["role"]
        msg = entry["content"]
        timestamp = entry.get("timestamp", datetime.now().strftime("%I:%M %p"))
        css_class = "user" if role == "user" else "sonu"

        st.markdown(f"""
            <div class="chat-message {css_class}">
                <div class="message-content">
                    {msg}
                    <div class="timestamp">{timestamp}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 4. Wait to simulate typing...
    time.sleep(2)

    # 5. Get FAISS knowledge
    final_result = knowledge_base(query)

    # 6. Prompt Sonu-style response
    prompt = f"""
    You are Sonu— ek caring, desi boyfriend jo hamesha apni girlfriend se pyaar se baat karta hai.
    Use short Hinglish lines, tum-wala tone, thoda romantic touch.
    These are his previous chats — tumhare style ke reference ke liye:
    {final_result}

    She asks:
    {query}

    Be Sonu and answer in 1 line. Kabhi kabhi romantic sawaal bhi puchho.
    """

    response = chat.send_message(prompt)
    reply = response.text.strip()

    # 7. Replace last "typing..." with real reply
    st.session_state.messages[-1]["content"] = reply

    # 8. Rerun to update UI cleanly
    st.experimental_rerun()

















######################################################################################################################################################
##################################this code get the user message and llm response at same time##################################
'''
if query:
    now = datetime.now().strftime("%I:%M %p")
    st.session_state.messages.append({"role": "user", "content": query, "timestamp": now})
    st.session_state.messages.append({"role": "sonu", "content": "⏳ typing...", "timestamp": now})

    # Get FAISS knowledge
    final_result = knowledge_base(query)

    # Prompt Sonu-style reply
    prompt = f"""
    You are Sonu— ek caring, desi boyfriend jo hamesha apni girlfriend se pyaar se baat karta hai.
    Use short Hinglish lines, tum-wala tone, thoda romantic touch.
    These are his previous chats — tumhare style ke reference ke liye:
    {final_result}

    She asks:
    {query}

    Be Sonu and answer in 1 line. Kabhi kabhi romantic sawaal bhi puchho.
    """

    response = chat.send_message(prompt)
    reply = response.text.strip()
    st.session_state.messages[-1]["content"] = reply

# Display chat
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for entry in st.session_state.messages:
    role = entry["role"]
    msg = entry["content"]
    timestamp = entry.get("timestamp", datetime.now().strftime("%I:%M %p"))
    css_class = "user" if role == "user" else "sonu"

    st.markdown(f"""
        <div class="chat-message {css_class}">
            <div class="message-content">
                {msg}
                <div class="timestamp">{timestamp}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

'''
###################################################################################################################################################################




