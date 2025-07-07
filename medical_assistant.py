import streamlit as st
import pandas as pd
import openai
from io import StringIO

# Configure page
st.set_page_config(
    page_title="Medical Results Interpreter",
    page_icon="🏥",
    layout="wide"
)

# Title and description
st.title("AI Medical Test Analyzer")
st.markdown("""
Upload patient test results in CSV format to get AI-powered interpretation and recommendations.
""")

# File upload section
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="Format: Test_Name,Result,Unit,Reference_Range"
)

if uploaded_file:
    # Read and display data
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File successfully uploaded!")
        
        with st.expander("View Raw Data"):
            st.dataframe(df)
        
        # Prepare data for analysis
        analysis_text = "\n".join(
            f"{row['Test_Name']}: {row['Result']} {row['Unit']} (Normal: {row['Reference_Range']})"
            for _, row in df.iterrows()
        )
        
        # Analysis button
        if st.button("Generate Medical Analysis", type="primary"):
            with st.spinner("Analyzing with AI..."):
                try:
                    # Initialize OpenAI with Streamlit secrets
                    openai.api_key = st.secrets["OPENAI_KEY"]
                    
                    # Generate analysis
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {
                                "role": "system",
                                "content": """You are a medical specialist. Analyze these test results:
                                1. Identify abnormal values
                                2. Explain potential health implications
                                3. Provide clinical recommendations
                                Use professional but patient-friendly language."""
                            },
                            {
                                "role": "user",
                                "content": analysis_text
                            }
                        ],
                        temperature=0.2  # Keep responses factual
                    )
                    
                    # Display results
                    analysis = response.choices[0].message.content
                    
                    st.subheader("AI Analysis Report")
                    st.markdown(f"""<div style='background:#f0f2f6;padding:20px;border-radius:10px'>
                                {analysis}
                                </div>""", unsafe_allow_html=True)
                    
                    # Add download option
                    st.download_button(
                        label="Download Full Report",
                        data=analysis,
                        file_name="medical_analysis.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.info("Please check your OpenAI key in Streamlit secrets")
    
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.info("Please ensure your CSV has columns: Test_Name,Result,Unit,Reference_Range")