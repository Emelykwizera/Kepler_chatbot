import pandas as pd
import streamlit as st
import re
import os
import traceback
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import hashlib

# ====================== CORE FUNCTIONS ======================
@st.cache_resource
def load_semantic_model():
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Failed to load AI model: {str(e)}")
        return None

def build_relationships(graph):
    try:
        all_entries = []
        for sheet in graph.values():
            all_entries.extend(sheet)
        
        for i, entry1 in enumerate(all_entries):
            for j, entry2 in enumerate(all_entries[i+1:], i+1):
                sim = cosine_similarity(
                    [entry1["embedding"]],
                    [entry2["embedding"]]
                )[0][0]
                
                if sim > 0.7:
                    entry1["related"].append((j, sim))
                    entry2["related"].append((i, sim))
    except Exception as e:
        st.warning(f"Couldn't build relationships: {str(e)}")

@st.cache_data
def load_data():
    try:
        excel_file = "kepler_data.xlsx"
        if not os.path.exists(excel_file):
            raise FileNotFoundError(f"Excel file not found at: {os.path.abspath(excel_file)}")
        
        sheets = {
            "Draft": None,
            "Admissions": None,
            "Orientation": None,
            "Programs": None
        }
        
        model = load_semantic_model()
        if model is None:
            return None, None
            
        knowledge_graph = defaultdict(list)
        
        for sheet_name in sheets.keys():
            try:
                df = pd.read_excel(excel_file, sheet_name=sheet_name, engine='openpyxl')
                
                if len(df.columns) < 2:
                    st.warning(f"Sheet '{sheet_name}' needs at least 2 columns, skipping...")
                    continue
                    
                df.columns = ["Questions", "Answers"] if len(df.columns) == 2 else [f"col_{i}" for i in range(len(df.columns))]
                
                questions = df["Questions"].astype(str).tolist()
                embeddings = model.encode(questions)
                
                for idx, row in df.iterrows():
                    entry = {
                        "question": str(row["Questions"]),
                        "answer": str(row["Answers"] if len(df.columns) >= 2 else ""),
                        "embedding": embeddings[idx],
                        "source": sheet_name,
                        "related": []
                    }
                    knowledge_graph[sheet_name].append(entry)
                
            except Exception as sheet_error:
                st.warning(f"Couldn't load sheet '{sheet_name}': {str(sheet_error)}")
                continue
        
        build_relationships(knowledge_graph)
        return sheets, knowledge_graph
        
    except Exception as e:
        st.error(f"Critical error loading data: {str(e)}")
        st.text(traceback.format_exc())
        return None, None

def semantic_search(user_question, knowledge_graph, model):
    try:
        if not user_question.strip():
            return None, None, ["Empty question provided"]
            
        question_embedding = model.encode([user_question])
        best_match = None
        best_score = 0
        best_source = None
        reasoning = []
        
        for sheet_name, entries in knowledge_graph.items():
            for entry in entries:
                sim = cosine_similarity(
                    [question_embedding[0]],
                    [entry["embedding"]]
                )[0][0]
                
                if sim > best_score:
                    best_score = sim
                    best_match = entry
                    best_source = sheet_name
                    reasoning = [f"Matched to: '{entry['question']}' (similarity: {sim:.2f})"]
                    
                    for rel_idx, rel_sim in entry["related"]:
                        rel_entry = None
                        for s in knowledge_graph.values():
                            if rel_idx < len(s):
                                rel_entry = s[rel_idx]
                                break
                            rel_idx -= len(s)
                        
                        if rel_entry and rel_sim > 0.6:
                            reasoning.append(f"Related concept: '{rel_entry['question']}' (similarity: {rel_sim:.2f})")
        
        threshold = max(0.5, 0.7 - (0.02 * len(user_question.split())))
        
        if best_score > threshold:
            answer = best_match["answer"]
            related_info = []
            
            for rel_idx, rel_sim in best_match["related"]:
                if rel_sim > 0.6:
                    rel_entry = None
                    for s in knowledge_graph.values():
                        if rel_idx < len(s):
                            rel_entry = s[rel_idx]
                            break
                        rel_idx -= len(s)
                    
                    if rel_entry:
                        related_info.append(f"\n\nRelated info: {rel_entry['answer']}")
            
            if related_info:
                answer += "\n".join(related_info[:2])
            
            return answer, best_source, reasoning
            
        return None, None, ["No sufficiently similar matches found"]
        
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return None, None, [f"Error during search: {str(e)}"]

def learn_from_interaction(user_question, response, knowledge_graph, model):
    try:
        if response and len(response) > 20:
            q_hash = hashlib.md5(user_question.encode()).hexdigest()[:8]
            new_entry = {
                "question": user_question,
                "answer": response,
                "embedding": model.encode([user_question])[0],
                "source": "learned",
                "related": []
            }
            
            if "learned" not in knowledge_graph:
                knowledge_graph["learned"] = []
            knowledge_graph["learned"].append(new_entry)
            build_relationships(knowledge_graph)
    except Exception as e:
        st.warning(f"Couldn't learn from interaction: {str(e)}")

# ====================== UI COMPONENTS ======================
def inject_custom_css():
    st.markdown("""
    <style>
        /* Developer image styling */
        .developer-image {
            position: fixed;
            top: 20px;
            left: 20px;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            object-fit: cover;
            border: 2px solid white;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            z-index: 1000;
        }
        
        /* Adjust header to account for image */
        .header {
            margin-top: 80px;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 2rem;
        }
        .chat-container {
            max-height: 60vh;
            overflow-y: auto;
            padding: 20px;
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            margin-bottom: 80px;
        }
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
            word-wrap: break-word;
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
            word-wrap: break-word;
        }
        .stChatInput > div {
            background: white !important;
            border-radius: 25px !important;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.05) !important;
        }
        .header h1 {
            color: #2563eb;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        @keyframes blink {
            0% { opacity: 0.2; }
            50% { opacity: 1; }
            100% { opacity: 0.2; }
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
        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }
        .error-message {
            color: #dc3545;
            background-color: #f8d7da;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
        }
    </style>
    """, unsafe_allow_html=True)

# ====================== MAIN APPLICATION ======================
def main():
    # Must be the first Streamlit command
    st.set_page_config(
        page_title="Kepler Thinking Assistant By Emely",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Inject CSS styles
    inject_custom_css()
    
    # Add your profile image (top-left corner)
    st.markdown("""
    <img src="mine.jpg" class="developer-image" title="Created by Emely">
    """, unsafe_allow_html=True)
    
    # Single header section
    st.markdown("""
    <div class="header">
        <h1>Kepler Thinking Assistant 🧠</h1>
        <p>Your intelligent guide to Kepler University</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load AI model and knowledge base
    with st.spinner("Loading AI model and knowledge base..."):
        model = load_semantic_model()
        data, knowledge_graph = load_data()
        
    if model is None or data is None:
        st.error("Failed to initialize required components. Please check the error messages above.")
        return
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.knowledge_graph = knowledge_graph
        st.session_state.typing = False
    
    # Display chat messages
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
        
        # Show typing indicator when thinking
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
    
    # Handle user input
    if prompt := st.chat_input("What would you like to know about Kepler?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.typing = True
        st.rerun()
        
        try:
            response, source, reasoning = semantic_search(prompt, st.session_state.knowledge_graph, model)
            
            if response:
                learn_from_interaction(prompt, response, st.session_state.knowledge_graph, model)
                formatted_response = f"{response}\n\n*(Source: {source} section)*"
                
                with st.expander("How I arrived at this answer"):
                    st.write("\n".join(reasoning))
            else:
                formatted_response = """
                I couldn't find a precise answer. You might ask about:
                - Admission requirements
                - Program details
                - Orientation schedules
                - Faculty information
                """
            
            st.session_state.messages.append({"role": "assistant", "content": formatted_response})
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "Sorry, I encountered an error processing your request. Please try again."
            })
        
        finally:
            st.session_state.typing = False
            st.rerun()

if __name__ == "__main__":
    main()