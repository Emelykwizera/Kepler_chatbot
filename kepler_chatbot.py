import pandas as pd
import streamlit as st
import re
import os
import base64
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ====================== CORE FUNCTIONS ======================
@st.cache_resource(show_spinner=False)
def load_semantic_model():
    try:
        # Pre-load the model with optimized settings
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        # Warmup with a small batch
        model.encode(["warmup"], batch_size=1, show_progress_bar=False)
        return model
    except Exception as e:
        st.error(f"Failed to load AI model: {str(e)}")
        return None

@st.cache_data(show_spinner=False, ttl=3600)
def load_data():
    try:
        excel_file = "kepler_data.xlsx"
        if not os.path.exists(excel_file):
            raise FileNotFoundError(f"Excel file not found at: {os.path.abspath(excel_file)}")
        
        model = load_semantic_model()
        if model is None:
            return None, None
            
        knowledge_graph = defaultdict(list)
        
        for sheet_name in ["Draft", "Admissions", "Orientation", "Programs"]:
            try:
                # Read data in optimized way
                df = pd.read_excel(excel_file, 
                                 sheet_name=sheet_name,
                                 usecols=["Questions", "Answers"],
                                 engine='openpyxl').dropna()
                
                questions = df["Questions"].astype(str).tolist()
                answers = df["Answers"].astype(str).tolist()
                
                # Process embeddings in batches for better performance
                batch_size = 32
                embeddings = []
                for i in range(0, len(questions), batch_size):
                    batch = questions[i:i + batch_size]
                    embeddings.extend(model.encode(batch, show_progress_bar=False, convert_to_tensor=False))
                
                for idx, (question, answer) in enumerate(zip(questions, answers)):
                    entry = {
                        "question": question,
                        "answer": answer,
                        "embedding": embeddings[idx],
                        "source": sheet_name,
                        "related": []
                    }
                    knowledge_graph[sheet_name].append(entry)
                
            except Exception as sheet_error:
                st.warning(f"Couldn't load sheet '{sheet_name}': {str(sheet_error)}")
                continue
        
        return {"loaded": True}, knowledge_graph
        
    except Exception as e:
        st.error(f"Critical error loading data: {str(e)}")
        return None, None

def semantic_search(user_question, knowledge_graph, model):
    try:
        if not user_question.strip():
            return None, None, []
            
        # Optimized question processing
        question_embedding = model.encode([user_question], 
                                        batch_size=1,
                                        show_progress_bar=False,
                                        convert_to_tensor=False)
        
        best_match = None
        best_score = 0
        best_source = None
        
        # Search priority based on likely user intent
        search_order = ["Admissions", "Programs", "Orientation", "Draft"]
        
        for sheet_name in search_order:
            for entry in knowledge_graph.get(sheet_name, []):
                sim = cosine_similarity(
                    [question_embedding],
                    [entry["embedding"]]
                )[0][0]
                
                if sim > 0.85:  # Early exit for good matches
                    return entry["answer"], sheet_name, [f"Exact match: {sim:.2f}"]
                
                if sim > best_score:
                    best_score = sim
                    best_match = entry["answer"]
                    best_source = sheet_name
        
        if best_score > 0.6:
            return best_match, best_source, [f"Best match: {best_score:.2f}"]
        return None, None, []

# ====================== UI COMPONENTS ====================== 
def inject_custom_css():
    st.markdown("""
    <style>
        /* Developer image */
        .developer-image {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            object-fit: cover;
            border: 2px solid #ffffff;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            cursor: pointer;
        }

        .developer-image:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.25);
        }

        
        /* Header */
        .header {
            margin-top: 70px;
            text-align: center;
            margin-bottom: 1rem;
        }
        
        /* Chat messages */
        .user-message {
            background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
            color: white;
            border-radius: 18px 18px 0 18px;
            padding: 12px 16px;
            margin: 8px 0;
            max-width: 80%;
            float: right;
            clear: both;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .bot-message {
            background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
            color: #333;
            border-radius: 18px 18px 18px 0;
            padding: 12px 16px;
            margin: 8px 0;
            max-width: 80%;
            float: left;
            clear: both;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .chat-container {
            max-height: 65vh;
            overflow-y: auto;
            padding: 15px;
        }
        .typing-indicator {
            display: inline-flex;
            padding: 0.5rem 1rem;
        }
        .typing-dot {
            animation: blink 1.4s infinite both;
            background-color: #666;
            border-radius: 50%;
            height: 8px;
            margin: 0 2px;
            width: 8px;
        }
        @keyframes blink {
            0% { opacity: 0.2; }
            50% { opacity: 1; }
            100% { opacity: 0.2; }
        }
    </style>
    """, unsafe_allow_html=True)

def get_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ====================== MAIN APPLICATION ======================
def main():
    st.set_page_config(
        page_title="Kepler Thinking Assistant",
        page_icon="🧠",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    inject_custom_css()
    
    # Load profile image (unchanged)
    try:
        img_base64 = get_image_base64("mine.jpg")
        st.markdown(f"""
        <div class="developer-image-container">
            <img src="data:image/jpeg;base64,{img_base64}" 
                 class="developer-image" 
                 title="Created by Emely">
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Couldn't load profile image: {str(e)}")
    
    # Header (unchanged)
    st.markdown("""
    <div class="header">
        <h1>Kepler Thinking Assistant 🧠</h1>
        <p>Ask me anything about Kepler University</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load resources with optimized caching
    with st.spinner("Initializing system (this only happens once)..."):
        model = load_semantic_model()
        _, knowledge_graph = load_data()
        
        if model is None or knowledge_graph is None:
            st.error("Failed to load required components")
            st.stop()
    
    # Initialize chat history (unchanged)
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.knowledge_graph = knowledge_graph
        st.session_state.typing = False
    
    # Display chat messages (unchanged)
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(
                    f'<div class="user-message">👤 {message["content"]}</div>', 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="bot-message">🤖 {message["content"]}</div>', 
                    unsafe_allow_html=True
                )
        
        if st.session_state.get("typing", False):
            st.markdown("""
            <div class="bot-message">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle user input (unchanged except for progress indicator)
    if prompt := st.chat_input("Type your question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.typing = True
        st.rerun()
        
        try:
            with st.spinner("🧠 Finding the best answer..."):
                response, source, _ = semantic_search(prompt, st.session_state.knowledge_graph, model)
            
            if response:
                formatted_response = f"{response}\n\n*(Source: {source} section)*"
            else:
                formatted_response = "I couldn't find a good answer. Try asking about admissions, programs, or orientation."
            
            st.session_state.messages.append({"role": "assistant", "content": formatted_response})
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "Sorry, I encountered an error. Please try again."
            })
        
        finally:
            st.session_state.typing = False
            st.rerun()

if __name__ == "__main__":
    main()