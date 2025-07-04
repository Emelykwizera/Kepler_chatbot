import pandas as pd
import streamlit as st
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import hashlib

# Set page configuration
st.set_page_config(
    page_title="Kepler Thinking Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load HTML template
def load_html_template():
    with open('index.html') as f:
        return f.read()

# Inject HTML template
html_template = load_html_template()
st.markdown(html_template, unsafe_allow_html=True)

# ===== CORE CHATBOT FUNCTIONALITY =====
# (Keep all your existing functions exactly as they were:
# load_semantic_model, load_data, build_relationships,
# semantic_search, learn_from_interaction, get_suggestions)

def main():
    # Initialize knowledge graph and model
    knowledge_graph = load_data()
    model = load_semantic_model()
    
    if knowledge_graph is None:
        st.error("Failed to load knowledge base")
        return
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.knowledge_graph = knowledge_graph
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input with suggestions
    prompt = st.chat_input("What would you like to know about Kepler?")
    
    if prompt:
        # Show suggestions
        suggestions = get_suggestions(prompt, st.session_state.knowledge_graph, model)
        selected_prompt = prompt
        
        if suggestions:
            with st.expander("💡 Similar questions"):
                for suggestion in suggestions:
                    if st.button(suggestion, key=f"suggest_{hashlib.md5(suggestion.encode()).hexdigest()}"):
                        selected_prompt = suggestion
        
        # Process user input
        st.session_state.messages.append({"role": "user", "content": selected_prompt})
        with st.chat_message("user"):
            st.markdown(selected_prompt)
        
        # Get bot response
        response, source, reasoning = semantic_search(
            selected_prompt, 
            st.session_state.knowledge_graph, 
            model
        )
        
        if response:
            learn_from_interaction(
                selected_prompt, 
                response, 
                st.session_state.knowledge_graph, 
                model
            )
            formatted_response = f"{response}\n\n*(Source: {source} section)*"
            
            with st.expander("How I arrived at this answer"):
                st.write("\n".join(reasoning))
        else:
            formatted_response = (
                "I'm not entirely sure about that. Try asking about:\n\n"
                "- Admission requirements\n"
                "- Program details\n"
                "- Orientation schedules"
            )
        
        # Display bot response
        with st.chat_message("assistant"):
            st.markdown(formatted_response)
        
        st.session_state.messages.append(
            {"role": "assistant", "content": formatted_response}
        )

if __name__ == "__main__":
    main()