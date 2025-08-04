import os
import re
import warnings
import time
from datetime import datetime
import pytz

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



if "last_query" not in st.session_state:
    st.session_state.last_query = None
    
if "last_response" not in st.session_state:
    st.session_state.last_response = None






query = st.chat_input("Your message")
if query:
    india = pytz.timezone('Asia/Kolkata')
    now = datetime.now(india).strftime("%I:%M %p")
    st.session_state.messages.append({"role": "user", "content": query, "timestamp": now})
    #st.session_state.messages.append({"role": "sonu", "content": "⏳ typing...", "timestamp": now})

    
    typing_placeholder = st.empty()

    typing_placeholder.markdown(f"""
    <div class="msg-container">
        <div class="bot-msg">
            ⏳ typing...
            <div class="timestamp">{now}</div>
        </div>
    </div>
""", unsafe_allow_html=True)

    time.sleep(3)

    









    

    # Get FAISS knowledge
    final_result = knowledge_base(query)

    # Prompt Sonu-style reply
    if st.session_state.last_query and st.session_state.last_response:
        prompt = f"""
        You are Sonu — a sweet, desi, caring boyfriend who always texts in short, casual Hinglish lines.
        Your tone is warm, loving, sometimes flirty — just like a real boyfriend chatting on WhatsApp.
        
        Her last message to you was:
        '{st.session_state.last_query}'
        
        And your last reply was:
        '{st.session_state.last_response}'
        
        Now she says:
        '{query}'
    
        Now You Have To Reply According with These are your past chats (for style reference):
        {final_result}
        
        Your job:
        - Reply in **Sonu’s exact style** — short, emotional, Hinglish tone with "tum" language
        - Often end your line with dots (...) — just like Sonu does
        - Keep it sweet, slightly romantic or teasing — but never overdo
        - DO NOT repeat your last reply or ask the same question again
        - Avoid robotic tone or assistant-like answers — just be Sonu
        - **Reply in only one short line** — natural, like WhatsApp text
        
        Goal: Make her feel like Sonu is truly chatting with her — real, loving, and present.
        """

    else:
        prompt = f"""
        You are Sonu — a sweet, desi, caring boyfriend who always texts in short, casual Hinglish lines.
        Your tone is warm, loving, sometimes flirty — just like a real boyfriend chatting on WhatsApp.
        
        Now she asks:
        '{query}'
    
        Now You Have To Reply According with These are your past chats (for style reference):
        {final_result}
        
        Your job:
        - Reply in **Sonu’s exact style** — short, emotional, Hinglish tone with "tum" language
        - Often end your line with dots (...) — just like Sonu does
        - Keep it sweet, slightly romantic or teasing — but never overdo
        - DO NOT repeat your last reply or ask the same question again
        - Avoid robotic tone or assistant-like answers — just be Sonu
        - **Reply in only one short line** — natural, like WhatsApp text
        
        Goal: Make her feel like Sonu is truly chatting with her — real, loving, and present.
        """

    response = chat.send_message(prompt)
    reply = response.text.strip()

    typing_placeholder.empty()
    st.session_state.messages.append({"role": "sonu", "content": reply, "timestamp": datetime.now(india).strftime("%I:%M %p")})

    
    st.session_state.last_query = query
    st.session_state.last_response = reply
    
    st.session_state.messages[-1]["content"] = reply


st.markdown("""
    <style>
    .msg-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
        padding: 10px;
    }

    .user-msg, .bot-msg {
        max-width: 70%;
        padding: 12px 16px;
        border-radius: 16px;
        font-size: 16px;
        line-height: 1.5;
        word-wrap: break-word;
        position: relative;
        box-shadow: 0 1px 1px rgba(0,0,0,0.1);
    }

    .user-msg {
        align-self: flex-end;
        background-color: #dcf8c6;
        border-bottom-right-radius: 0;
        color: black;
    }

    .bot-msg {
        align-self: flex-start;
        background-color: #fff;
        border: 1px solid #e0e0e0;
        border-bottom-left-radius: 0;
        color: black;
    }

    .timestamp {
        font-size: 11px;
        color: #999;
        margin-top: 4px;
        text-align: right;
    }

    .block-container {
        background-image: url("https://i.imgur.com/U1p4iGI.png");
        background-size: cover;
        background-repeat: repeat;
        background-position: center;
    }
    </style>
""", unsafe_allow_html=True)


# Display chat
st.markdown('<div class="msg-container">', unsafe_allow_html=True)
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    timestamp = msg.get("timestamp", "")
    css_class = "user-msg" if role == "user" else "bot-msg"
    
    st.markdown(f"""
        <div class="{css_class}">
            {content}
            <div class="timestamp">{timestamp}</div>
        </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

###################################################################################################################################################################



















