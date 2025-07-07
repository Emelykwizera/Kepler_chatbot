import streamlit as st
import pandas as pd
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Package Validation ---
try:
    import openai
except ImportError as e:
    st.error(f"Critical Error: {str(e)}")
    st.markdown("""
    **Solution:**  
    Please ensure your `requirements.txt` contains:
    ```text
    openai>=1.0.0
    pandas>=2.0.0
    ```
    """)
    st.stop()

# --- API Key Validation ---
if 'OPENAI_KEY' not in st.secrets:
    st.error("""
    **API Key Missing**  
    Add your OpenAI key in Streamlit Secrets:
    1. Go to app settings (⚙ icon)
    2. Select "Secrets"
    3. Add:  
    ```toml
    OPENAI_KEY = "your-api-key-here"
    ```
    """)
    st.stop()

# Initialize OpenAI
client = openai.OpenAI(api_key=st.secrets.OPENAI_KEY)

# --- Streamlit App ---
st.set_page_config(
    page_title="Medical AI Analyzer",
    page_icon="🏥",
    layout="wide"
)

def analyze_results(df):
    """Core analysis function with error handling"""
    try:
        analysis_text = "\n".join(
            f"{row['Test_Name']}: {row['Result']} {row['Unit']} (Normal: {row['Reference_Range']})"
            for _, row in df.iterrows()
        )
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a medical expert. Analyze test results professionally."},
                {"role": "user", "content": analysis_text}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return f"Analysis Error: {str(e)}"

# --- UI Components ---
st.title("Medical Test Interpreter")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File loaded successfully!")
        
        with st.expander("View Data"):
            st.dataframe(df)
        
        if st.button("Analyze Results", type="primary"):
            with st.spinner("Generating medical analysis..."):
                analysis = analyze_results(df)
                st.subheader("Medical Analysis")
                st.markdown(f"""<div style='background:#f0f9ff;padding:20px;border-radius:10px'>
                            {analysis}
                            </div>""", unsafe_allow_html=True)
                
                st.download_button(
                    "Download Report",
                    analysis,
                    file_name="medical_report.txt"
                )
                
    except Exception as e:
        st.error(f"File Error: {str(e)}")
        st.markdown("""
        **Required CSV Format:**
        ```csv
        Test_Name,Result,Unit,Reference_Range
        Glucose,98,mg/dL,70-99
        HbA1c,5.8,%,<5.7
        ```
        """)
