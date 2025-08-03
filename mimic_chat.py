
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
#import preprocess
import re
import google.generativeai as genai
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime

model_embed = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

load_dotenv() 
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")
chat= model.start_chat()


class EmbeddingModel(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)

    def embed_query(self, text):
        return self.model.encode(text, convert_to_numpy=True)

wrapped_model = EmbeddingModel(model_embed)


def clean_chat(file_path):
    # Pattern to match lines like: [12/12/2023, 10:29:22 PM] Sonu: Message
    pattern = re.compile(r"^\[\d{1,2}/\d{1,2}/\d{4}, \d{1,2}:\d{2}:\d{2}.*?\] ")

    messages = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if pattern.match(line):
                try:
                    # Remove timestamp block
                    line_clean = re.sub(r"^\[.*?\] ", "", line).strip()
                    # Split sender and message
                    if ':' in line_clean:
                        sender, message = line_clean.split(":", 1)
                        messages.append({
                            "sender": sender.strip(),
                            "text": message.strip()
                        })
                except ValueError:
                    continue
    return messages



#
def data_make(file_path):
    messages = clean_chat(file_path)
    sonu_lines = [msg["text"] for msg in messages if "sonu" in msg["sender"].lower()]
    documents = [Document(page_content=text) for text in sonu_lines]
    return documents


def faiss_database(documents):
    if os.path.exists("faiss_index_chat"):
        print("already exist: Loading....")
        loaded_faiss=FAISS.load_local("faiss_index_chat", embeddings=wrapped_model, allow_dangerous_deserialization=True)
        

    else:
        print("Making new faiss database")
        faiss_index_chat = FAISS.from_documents(documents, embedding=wrapped_model)
        faiss_index_chat.save_local("faiss_index_chat")
        loaded_faiss=FAISS.load_local("faiss_index_chat", embeddings=wrapped_model, allow_dangerous_deserialization=True)

    retriever = loaded_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    return retriever




def load_faiss_index():
    documents = data_make("chat.txt")  
    retriever = faiss_database(documents)  
    return retriever 

retriever= load_faiss_index()

def knowledge_base(query):
    results = retriever.invoke(query)
    return results


##############################################################################################
# Bubble-style CSS using flex layout
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
        color: black;  /* Set text color to black */
    }
    .sonu {
        justify-content: flex-start;
    }
    .sonu .message-content {
        background-color: #ffffff;
        border: 1px solid #ccc;
        border-bottom-left-radius: 0px;
        color: black;  /* Set text color to black */
    }
    .timestamp {
        font-size: 10px;
        color: #999;
        margin-top: 2px;
        text-align: right;
    }
    </style>
""", unsafe_allow_html=True)
#############################################################################################

st.markdown('<div class="chat-container">', unsafe_allow_html=True)



if "messages" not in st.session_state:
    st.session_state.messages = []



query= st.chat_input("your message")

if query: 
    now = datetime.now().strftime("%I:%M %p")
    st.session_state.messages.append({"role": "user","content": query,"timestamp": now})
    placeholder = st.empty()

    st.session_state.messages.append({"role": "sonu","content": "⏳ typing...","timestamp": now})
 
    final_result= knowledge_base(query)
    
    prompt = f"""
    You are Sonu— ek caring, desi boyfriend jo hamesha apni girlfriend se pyaar se baat karta hai.
    Use short Hinglish lines, tum-wala tone, thoda romantic touch.
    these are him previous chats — tumhare style ke reference ke liye:
    {final_result}

    she asks:
    "{query}"

    Be Sonu and answer in 1 line. "Kabhi kabhi romantic sawaal bhi puchho."
    """

    response = chat.send_message(prompt)
    reply= response.text.strip()
    
    #st.session_state.messages.append({"role": "sonu","content": reply,"timestamp": datetime.now().strftime("%I:%M %p")})
    st.session_state.messages[-1]["content"] = reply



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








   # with st.chat_message("sonu"):
   #     st.markdown(reply)

        #st.experimental_rerun()
        #st.write(reply)
    