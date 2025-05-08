import streamlit as st
import requests
import pandas as pd
import json

# Configure the app
st.set_page_config(
    page_title="NLP Preprocessing Demo",
    page_icon="🔤",
    layout="wide"
)

# API endpoint URL
API_URL = "http://localhost:8000"

def make_api_request(endpoint, text):
    """Make a request to the API endpoint"""
    url = f"{API_URL}/{endpoint}"
    try:
        response = requests.post(url, json={"text": text}, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"API request failed: {e}")
        return None

# App header
st.title("NLP Preprocessing Demo")
st.markdown("""
This app demonstrates NLP preprocessing techniques using the FastAPI backend.
Enter your text below and select a preprocessing function to see the results.
""")

# Text input
text_input = st.text_area(
    "Enter text to process:",
    "Natural Language Processing (NLP) is a subfield of artificial intelligence. It helps computers understand, interpret, and manipulate human language. The goal of NLP is to bridge the gap between human communication and computer understanding. Dr. Ram developed a new algorithm at Kathmandu University.",
    height=150
)

# Select function
preprocessing_function = st.selectbox(
    "Select preprocessing function:",
    ["Tokenization", "Lemmatization", "Stemming", "POS Tagging", "Named Entity Recognition", "All"]
)

# Process button
if st.button("Process"):
    if not text_input:
        st.warning("Please enter some text first.")
    else:
        # Show spinner while processing
        with st.spinner("Processing..."):
            endpoint_map = {
                "Tokenization": "tokenize",
                "Lemmatization": "lemmatize",
                "Stemming": "stem",
                "POS Tagging": "pos-tag",
                "Named Entity Recognition": "ner",
                "All": "process-all"
            }
            
            endpoint = endpoint_map[preprocessing_function]
            result = make_api_request(endpoint, text_input)
            
            if result:
                # Display results based on the selected function
                if preprocessing_function == "Tokenization":
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("NLTK Tokenization")
                        st.write("**Sentences:**")
                        for i, sentence in enumerate(result["nltk"]["sentences"], 1):
                            st.write(f"{i}. {sentence}")
                        st.write("**Words:**")
                        st.write(result["nltk"]["words"])
                    
                    with col2:
                        st.subheader("spaCy Tokenization")
                        st.write("**Sentences:**")
                        for i, sentence in enumerate(result["spacy"]["sentences"], 1):
                            st.write(f"{i}. {sentence}")
                        st.write("**Tokens:**")
                        st.write(result["spacy"]["tokens"])
                
                elif preprocessing_function == "Lemmatization":
                    st.subheader("Lemmatization Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**NLTK Lemmatization:**")
                        if result["nltk_pairs"]:
                            st.table(pd.DataFrame(result["nltk_pairs"]))
                        else:
                            st.write("No words were changed by NLTK lemmatization.")
                    
                    with col2:
                        st.write("**spaCy Lemmatization:**")
                        if result["spacy_pairs"]:
                            st.table(pd.DataFrame(result["spacy_pairs"]))
                        else:
                            st.write("No words were changed by spaCy lemmatization.")
                
                elif preprocessing_function == "Stemming":
                    st.subheader("Stemming Results")
                    st.write("Comparison of stemming algorithms:")
                    st.table(pd.DataFrame(result["comparison"]))
                
                elif preprocessing_function == "POS Tagging":
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("NLTK POS Tagging")
                        st.table(pd.DataFrame(result["nltk"]))
                    
                    with col2:
                        st.subheader("spaCy POS Tagging")
                        st.table(pd.DataFrame(result["spacy"]))
                
                elif preprocessing_function == "Named Entity Recognition":
                    st.subheader("Named Entity Recognition")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**NLTK Named Entities:**")
                        if result["nltk"]:
                            st.table(pd.DataFrame(result["nltk"]))
                        else:
                            st.write("No named entities detected by NLTK.")
                    
                    with col2:
                        st.write("**spaCy Named Entities:**")
                        if result["spacy"]:
                            st.table(pd.DataFrame(result["spacy"]))
                        else:
                            st.write("No named entities detected by spaCy.")
                
                elif preprocessing_function == "All":
                    tabs = st.tabs(["Tokenization", "Lemmatization", "Stemming", "POS Tagging", "NER"])
                    
                    # Tokenization tab
                    with tabs[0]:
                        tokenization = result["tokenization"]
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("NLTK Tokenization")
                            st.write("**Sentences:**")
                            for i, sentence in enumerate(tokenization["nltk"]["sentences"], 1):
                                st.write(f"{i}. {sentence}")
                            st.write("**Words:**")
                            st.write(tokenization["nltk"]["words"])
                        
                        with col2:
                            st.subheader("spaCy Tokenization")
                            st.write("**Sentences:**")
                            for i, sentence in enumerate(tokenization["spacy"]["sentences"], 1):
                                st.write(f"{i}. {sentence}")
                            st.write("**Tokens:**")
                            st.write(tokenization["spacy"]["tokens"])
                    
                    # Lemmatization tab
                    with tabs[1]:
                        lemmatization = result["lemmatization"]
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("NLTK Lemmatization")
                            if lemmatization["nltk_pairs"]:
                                st.table(pd.DataFrame(lemmatization["nltk_pairs"]))
                            else:
                                st.write("No words were changed by NLTK lemmatization.")
                        
                        with col2:
                            st.subheader("spaCy Lemmatization")
                            if lemmatization["spacy_pairs"]:
                                st.table(pd.DataFrame(lemmatization["spacy_pairs"]))
                            else:
                                st.write("No words were changed by spaCy lemmatization.")
                    
                    # Stemming tab
                    with tabs[2]:
                        stemming = result["stemming"]
                        st.subheader("Stemming Results")
                        st.table(pd.DataFrame(stemming["comparison"]))
                    
                    # POS Tagging tab
                    with tabs[3]:
                        pos_tagging = result["pos_tagging"]
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("NLTK POS Tagging")
                            st.table(pd.DataFrame(pos_tagging["nltk"]))
                        
                        with col2:
                            st.subheader("spaCy POS Tagging")
                            st.table(pd.DataFrame(pos_tagging["spacy"]))
                    
                    # NER tab
                    with tabs[4]:
                        ner = result["ner"]
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("NLTK Named Entities")
                            if ner["nltk"]:
                                st.table(pd.DataFrame(ner["nltk"]))
                            else:
                                st.write("No named entities detected by NLTK.")
                        
                        with col2:
                            st.subheader("spaCy Named Entities")
                            if ner["spacy"]:
                                st.table(pd.DataFrame(ner["spacy"]))
                            else:
                                st.write("No named entities detected by spaCy.")