import pandas as pd
import streamlit as st
import re
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import hashlib

# Set page config
st.set_page_config(page_title="Kepler Thinking Chatbot by Emely", page_icon="🧠")

# Load custom CSS (corrected URL format)
css_url = "https://raw.githubusercontent.com/Emelykwizera/Kepler_chatbot/main/style.css"
st.markdown(f'<link rel="stylesheet" href="{css_url}">', unsafe_allow_html=True)

# Set logo
st.image("mine.jpg", width=100)  # Ensure this image is in your repo

# Load pre-trained semantic model
@st.cache_resource
def load_semantic_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Suggestion function (NEW)
def get_suggestions(user_input, knowledge_graph, model, top_n=5):
    """Returns top N similar questions from the knowledge base"""
    if len(user_input) < 3:  # Don't search for very short inputs
        return []
    
    input_embedding = model.encode([user_input])[0]
    all_questions = []
    
    for sheet in knowledge_graph.values():
        for entry in sheet:
            all_questions.append((entry["question"], entry["embedding"]))
    
    similarities = []
    for q, emb in all_questions:
        sim = cosine_similarity([input_embedding], [emb])[0][0]
        similarities.append((q, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [q for q, sim in similarities[:top_n] if sim > 0.5]

# Enhanced data loader with semantic embeddings
@st.cache_data
def load_data():
    excel_file = "kepler_data.xlsx"
    sheets = {
        "Draft": None,
        "Admissions": None,
        "Orientation": None,
        "Programs": None
    }
    
    try:
        model = load_semantic_model()
        knowledge_graph = defaultdict(list)
        
        for sheet in sheets.keys():
            df = pd.read_excel(excel_file, sheet_name=sheet)
            df.columns = ["Questions", "Answers"]
            
            questions = df["Questions"].astype(str).tolist()
            embeddings = model.encode(questions)
            
            for idx, row in df.iterrows():
                entry = {
                    "question": str(row["Questions"]),
                    "answer": str(row["Answers"]),
                    "embedding": embeddings[idx],
                    "source": sheet,
                    "related": []
                }
                knowledge_graph[sheet].append(entry)
            
            sheets[sheet] = df
        
        build_relationships(knowledge_graph)
        return sheets, knowledge_graph
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def build_relationships(graph):
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

def semantic_search(user_question, knowledge_graph, model):
    question_embedding = model.encode([user_question])
    best_match = None
    best_score = 0
    best_source = None
    reasoning = []
    
    for sheet, entries in knowledge_graph.items():
        for entry in entries:
            sim = cosine_similarity(
                [question_embedding[0]],
                [entry["embedding"]]
            )[0][0]
            
            if sim > best_score:
                best_score = sim
                best_match = entry
                best_source = sheet
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
            answer += "\n\n" + "\n".join(related_info[:2])
        
        return answer, best_source, reasoning
    
    return None, None, []

def learn_from_interaction(user_question, response, knowledge_graph, model):
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

def main():
    st.title("Kepler Thinking Assistant 🧠")
    st.write("Ask me anything - I understand context and make connections!")
    
    data, knowledge_graph = load_data()
    model = load_semantic_model()
    
    if data is None:
        return
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.knowledge_graph = knowledge_graph
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Modified chat input with suggestions
    prompt = st.chat_input("What would you like to know about Kepler?")
    
    if prompt:
        # Show suggestions
        suggestions = get_suggestions(prompt, st.session_state.knowledge_graph, model)
        selected_prompt = prompt  # Default to original input
        
        if suggestions:
            with st.expander("💡 Did you mean one of these?"):
                for suggestion in suggestions:
                    if st.button(suggestion, use_container_width=True, 
                               key=f"suggest_{hashlib.md5(suggestion.encode()).hexdigest()}"):
                        selected_prompt = suggestion
        
        # Process the selected prompt
        st.session_state.messages.append({"role": "user", "content": selected_prompt})
        with st.chat_message("user"):
            st.markdown(selected_prompt)
        
        response, source, reasoning = semantic_search(selected_prompt, st.session_state.knowledge_graph, model)
        
        if response:
            learn_from_interaction(selected_prompt, response, st.session_state.knowledge_graph, model)
            formatted_response = f"{response}\n\n*(Source: {source} section)*"
            
            with st.expander("How I arrived at this answer"):
                st.write("\n".join(reasoning))
        else:
            formatted_response = ("I'm not entirely sure about that. Based on what I know, "
                               "you might want to ask about:\n\n"
                               "- Admission requirements\n"
                               "- Program details\n"
                               "- Orientation schedules\n\n"
                               "Could you rephrase or ask about one of these areas?")
        
        with st.chat_message("assistant"):
            st.markdown(formatted_response)
        
        st.session_state.messages.append({"role": "assistant", "content": formatted_response})

if __name__ == "__main__":
    main()