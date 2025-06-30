import pandas as pd
import streamlit as st
import re
import os
import traceback
import base64
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
from typing import Tuple, Optional, Dict, List

# ====================== CONSTANTS & CONFIGURATION ======================
DATA_FILE = "kepler_data.xlsx"
SHEETS_TO_LOAD = ["Admissions", "Programs", "Orientation", "Draft"]
MODEL_NAME = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.85
MIN_SIMILARITY = 0.6

# ====================== LOGGING SETUP ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ====================== CORE FUNCTIONS ======================
@st.cache_resource(show_spinner=False)
def load_semantic_model() -> Optional[SentenceTransformer]:
    """Load and warm up the semantic similarity model."""
    try:
        model = SentenceTransformer(MODEL_NAME, device='cpu')
        model.encode(["warmup"], show_progress_bar=False)
        logger.info("Semantic model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load AI model: {str(e)}")
        st.error("⚠️ Failed to initialize the AI engine. Please try again later.")
        return None

@st.cache_data(show_spinner=False, ttl=3600)
def load_data() -> Tuple[Optional[Dict], Optional[Dict]]:
    """Load and process the knowledge base from Excel."""
    try:
        if not os.path.exists(DATA_FILE):
            raise FileNotFoundError(f"Data file not found at: {os.path.abspath(DATA_FILE)}")
        
        model = load_semantic_model()
        if model is None:
            return None, None
            
        knowledge_graph = defaultdict(list)
        
        for sheet_name in SHEETS_TO_LOAD:
            try:
                df = pd.read_excel(
                    DATA_FILE,
                    sheet_name=sheet_name,
                    usecols=["Questions", "Answers"],
                    engine='openpyxl'
                ).dropna()
                
                questions = df["Questions"].astype(str).tolist()
                answers = df["Answers"].astype(str).tolist()
                
                if not questions:
                    logger.warning(f"No questions found in sheet: {sheet_name}")
                    continue
                
                embeddings = model.encode(questions, show_progress_bar=False)
                
                for idx, (question, answer) in enumerate(zip(questions, answers)):
                    entry = {
                        "question": question,
                        "answer": answer,
                        "embedding": embeddings[idx],
                        "source": sheet_name,
                        "related": []
                    }
                    knowledge_graph[sheet_name].append(entry)
                
                logger.info(f"Successfully loaded sheet: {sheet_name}")
                
            except Exception as sheet_error:
                logger.warning(f"Couldn't load sheet '{sheet_name}': {str(sheet_error)}")
                continue
        
        return {"loaded": True}, knowledge_graph
        
    except Exception as e:
        logger.error(f"Critical error loading data: {str(e)}")
        st.error("⚠️ Failed to load knowledge base. Please check the data file.")
        return None, None

def semantic_search(
    user_question: str,
    knowledge_graph: Dict,
    model: SentenceTransformer
) -> Tuple[Optional[str], Optional[str], List[str]]:
    """Perform semantic search on the knowledge graph."""
    try:
        if not user_question.strip():
            return None, None, []
            
        # Pre-process the question
        sanitized_question = re.sub(r'[^\w\s]', '', user_question.lower())
        question_embedding = model.encode([sanitized_question], show_progress_bar=False)
        
        best_match = None
        best_score = 0
        best_source = None
        diagnostics = []
        
        # Search through all sheets in priority order
        for sheet_name in SHEETS_TO_LOAD:
            for entry in knowledge_graph.get(sheet_name, []):
                sim = cosine_similarity(
                    question_embedding,
                    [entry["embedding"]]
                )[0][0]
                
                if sim > SIMILARITY_THRESHOLD:
                    return entry["answer"], sheet_name, [f"Exact match: {sim:.2f}"]
                
                if sim > best_score:
                    best_score = sim
                    best_match = entry["answer"]
                    best_source = sheet_name
        
        if best_score > MIN_SIMILARITY:
            diagnostics.append(f"Best match: {best_score:.2f}")
            return best_match, best_source, diagnostics
        
        return None, None, []

    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        return None, None, ["Search error"]

# ====================== UI COMPONENTS ======================
def inject_custom_css() -> None:
    """Inject custom CSS styles for the application."""
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
        
        /* Suggested questions */
        .suggested-question {
            display: inline-block;
            margin: 0.25rem;
            padding: 0.5rem 1rem;
            background-color: #f0f2f6;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .suggested-question:hover {
            background-color: #dbe4f8;
            transform: scale(1.02);
        }
    </style>
    """, unsafe_allow_html=True)

def get_image_base64(path: str) -> str:
    """Convert image to base64 string."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception as e:
        logger.warning(f"Couldn't load image: {str(e)}")
        return ""

def format_response(response: str, source: str) -> str:
    """Format the bot response with consistent styling."""
    if not response:
        return "I couldn't find a good answer. Try asking about:\n- Admissions requirements\n- Available programs\n- Orientation schedule"
    
    return f"""
{response}

---
*Source: {source} | [✉️ Report issue](#)*
"""

def get_suggested_questions(topic: str) -> List[str]:
    """Get contextually relevant follow-up questions."""
    suggestions = {
        "Admissions": [
            "What are the admission requirements?",
            "When is the application deadline?",
            "What documents do I need to apply?"
        ],
        "Programs": [
            "What undergraduate programs are offered?",
            "Are there any scholarship opportunities?",
            "What's the duration of the MBA program?"
        ],
        "Orientation": [
            "When does orientation week start?",
            "What activities are planned for orientation?",
            "Is orientation mandatory for new students?"
        ],
        "default": [
            "Tell me about admissions",
            "What programs are available?",
            "When is the next orientation?"
        ]
    }
    return suggestions.get(topic, suggestions["default"])

# ====================== MAIN APPLICATION ======================
def initialize_session_state() -> None:
    """Initialize the Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.knowledge_graph = None
        st.session_state.typing = False
        st.session_state.last_topic = None
        st.session_state.suggested_questions = []

def display_header() -> None:
    """Display the application header and profile image."""
    try:
        img_base64 = get_image_base64("mine.jpg")
        st.markdown(f"""
        <div style="text-align: center;">
            <img src="data:image/jpeg;base64,{img_base64}" 
                 class="developer-image" 
                 title="Created by Emely">
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        logger.warning(f"Couldn't load profile image: {str(e)}")
    
    st.markdown("""
    <div class="header">
        <h1>Kepler Thinking Assistant 🧠</h1>
        <p>Ask me anything about Kepler University</p>
    </div>
    """, unsafe_allow_html=True)

def display_chat() -> None:
    """Display the chat conversation and typing indicator."""
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

def display_suggested_questions() -> None:
    """Display suggested follow-up questions if available."""
    if st.session_state.suggested_questions:
        st.markdown("**You might ask:**")
        cols = st.columns(3)
        for i, question in enumerate(st.session_state.suggested_questions[:3]):
            with cols[i % 3]:
                if st.button(question, key=f"suggested_{i}"):
                    handle_user_question(question)

def handle_user_question(question: str) -> None:
    """Process a user question and generate response."""
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.typing = True
    st.rerun()
    
    try:
        response, source, _ = semantic_search(
            question,
            st.session_state.knowledge_graph,
            load_semantic_model()
        )
        
        formatted_response = format_response(response, source)
        st.session_state.messages.append({
            "role": "assistant",
            "content": formatted_response
        })
        
        # Update suggested questions based on topic
        st.session_state.last_topic = source if source else "default"
        st.session_state.suggested_questions = get_suggested_questions(
            st.session_state.last_topic
        )
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "⚠️ Sorry, I encountered an error. Please try again."
        })
        st.session_state.suggested_questions = get_suggested_questions("default")
    finally:
        st.session_state.typing = False
        st.rerun()

def main() -> None:
    """Main application function."""
    st.set_page_config(
        page_title="Kepler Thinking Assistant",
        page_icon="🧠",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    inject_custom_css()
    initialize_session_state()
    display_header()
    
    # Load resources
    with st.spinner("Initializing system..."):
        if st.session_state.knowledge_graph is None:
            _, knowledge_graph = load_data()
            if knowledge_graph is not None:
                st.session_state.knowledge_graph = knowledge_graph
            else:
                st.error("Failed to load knowledge base. Please check the data file.")
                st.stop()
    
    display_chat()
    display_suggested_questions()
    
    # Handle user input
    if prompt := st.chat_input("Type your question..."):
        handle_user_question(prompt)

if __name__ == "__main__":
    main()