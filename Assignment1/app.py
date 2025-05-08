import streamlit as st
import requests
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Configure the app
st.set_page_config(
    page_title="NLP Processing Demo",
    page_icon="🔤",
    layout="wide"
)

# API endpoint URL
API_URL = "http://localhost:8000"

def make_api_request(endpoint, payload):
    """Make a request to the API endpoint"""
    url = f"{API_URL}/{endpoint}"
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"API request failed: {e}")
        return None

def make_get_request(endpoint):
    """Make a GET request to the API endpoint"""
    url = f"{API_URL}/{endpoint}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"API request failed: {e}")
        return None

# App title
st.title("NLP Processing & Embeddings Demo")

# Create tabs for different functionality
tab1, tab2 = st.tabs(["Text Preprocessing", "Word Embeddings"])

# Tab 1: Text Preprocessing
with tab1:
    st.header("Text Preprocessing")
    st.markdown("""
    This section demonstrates NLP preprocessing techniques using the FastAPI backend.
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
    if st.button("Process Text"):
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
                result = make_api_request(endpoint, {"text": text_input})
                
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
                        all_tabs = st.tabs(["Tokenization", "Lemmatization", "Stemming", "POS Tagging", "NER"])
                        
                        # Tokenization tab
                        with all_tabs[0]:
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
                        with all_tabs[1]:
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
                        with all_tabs[2]:
                            stemming = result["stemming"]
                            st.subheader("Stemming Results")
                            st.table(pd.DataFrame(stemming["comparison"]))
                        
                        # POS Tagging tab
                        with all_tabs[3]:
                            pos_tagging = result["pos_tagging"]
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("NLTK POS Tagging")
                                st.table(pd.DataFrame(pos_tagging["nltk"]))
                            
                            with col2:
                                st.subheader("spaCy POS Tagging")
                                st.table(pd.DataFrame(pos_tagging["spacy"]))
                        
                        # NER tab
                        with all_tabs[4]:
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

# Tab 2: Word Embeddings
with tab2:
    st.header("Word Embeddings")
    st.markdown("""
    This section lets you explore word embeddings and find similar words.
    Enter words to see their embeddings and their most similar words based on cosine similarity.
    """)
    
    # Get available vocabulary for display
    vocab_data = make_get_request("vocabulary")
    if vocab_data:
        st.info(f"Vocabulary size: {vocab_data['vocabulary_size']} words")
        
        with st.expander("Show sample vocabulary words"):
            st.write(vocab_data['words'])
    
    # Create subtabs for different embedding functionalities
    embed_tab1, embed_tab2 = st.tabs(["Single Word Lookup", "Compare Multiple Words"])
    
    # Single Word Lookup tab
    with embed_tab1:
        st.subheader("Find Similar Words")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            word_input = st.text_input("Enter a word:", "technology")
            num_neighbors = st.slider("Number of neighbors:", 1, 10, 5)
            lookup_button = st.button("Find Similar Words")
        
        if lookup_button and word_input:
            with st.spinner("Looking up word..."):
                similar_words_data = make_api_request("get-similar-words", {"word": word_input})
                
                if similar_words_data:
                    with col2:
                        st.success(f"Found similar words for '{similar_words_data['word']}'")
                        
                        # Create a dataframe for the similar words
                        similar_df = pd.DataFrame([
                            {"Word": item["word"], "Similarity": f"{item['similarity']:.4f}"} 
                            for item in similar_words_data["similar_words"][:num_neighbors]
                        ])
                        
                        st.table(similar_df)
                    
                    # Also show the embedding vector
                    embedding_data = make_api_request("get-word-embedding", {"word": word_input})
                    if embedding_data:
                        with st.expander("Show embedding vector"):
                            st.write(f"Dimension: {embedding_data['dimension']}")
                            embedding_array = np.array(embedding_data['embedding'])
                            
                            # Display first few values
                            st.write("First 10 values of the embedding vector:")
                            st.write(embedding_array[:10])
                            
                            # Visualize the embedding
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.plot(embedding_array)
                            ax.set_title(f"Embedding Vector for '{embedding_data['word']}'")
                            ax.set_xlabel("Dimension")
                            ax.set_ylabel("Value")
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
    
    # Compare Multiple Words tab
    with embed_tab2:
        st.subheader("Compare Multiple Words")
        
        words_input = st.text_area(
            "Enter words (one per line):", 
            "technology\nscience\nart\nsport"
        )
        
        compare_col1, compare_col2 = st.columns([1, 1])
        
        with compare_col1:
            num_neighbors = st.slider("Neighbors per word:", 1, 5, 3, key="multi_neighbors")
            viz_method = st.radio("Visualization method:", ["PCA", "t-SNE"])
        
        with compare_col2:
            compare_button = st.button("Compare Words")
            st.markdown("""
            **Note:** Enter at least 2 words for visualization. 
            Words not found in the vocabulary will be ignored.
            """)
        
        if compare_button and words_input:
            words_list = [w.strip() for w in words_input.split("\n") if w.strip()]
            
            if not words_list:
                st.warning("Please enter at least one word.")
            else:
                with st.spinner("Comparing words..."):
                    multiple_data = make_api_request(
                        "get-multiple-embeddings", 
                        {"words": words_list, "num_neighbors": num_neighbors}
                    )
                    
                    if multiple_data:
                        # Extract valid results (words found in vocabulary)
                        valid_results = [r for r in multiple_data["results"] if r.get("in_vocabulary", False)]
                        invalid_words = [r["word"] for r in multiple_data["results"] if not r.get("in_vocabulary", False)]
                        
                        if invalid_words:
                            st.warning(f"Words not found in vocabulary: {', '.join(invalid_words)}")
                        
                        if not valid_results:
                            st.error("None of the entered words were found in the vocabulary.")
                        else:
                            # Display similar words for each valid word
                            st.markdown(f"### Found {len(valid_results)} words in vocabulary")
                            
                            # Create a tab for each valid word
                            if len(valid_results) > 0:
                                word_tabs = st.tabs([r["word"] for r in valid_results])
                                
                                for i, result in enumerate(valid_results):
                                    with word_tabs[i]:
                                        similar_words = result["similar_words"]
                                        if similar_words:
                                            similar_df = pd.DataFrame([
                                                {"Word": item["word"], "Similarity": f"{item['similarity']:.4f}"} 
                                                for item in similar_words
                                            ])
                                            st.table(similar_df)
                                        else:
                                            st.write("No similar words found.")
                            
                            # Collect embeddings for visualization
                            labels = [r["word"] for r in valid_results]
                            embeddings = np.array([r["embedding"] for r in valid_results])
                            
                            if len(embeddings) >= 2:  # Need at least 2 points for visualization
                                st.markdown("### Visualization of Word Embeddings")
                                
                                # Perform dimensionality reduction
                                if viz_method == "PCA":
                                    pca = PCA(n_components=2)
                                    reduced_data = pca.fit_transform(embeddings)
                                    title = "PCA Visualization of Word Embeddings"
                                else:  # t-SNE
                                    # Use smaller perplexity for small datasets to avoid warnings
                                    perplexity_value = min(3, len(embeddings) - 1) if len(embeddings) < 5 else 5
                                    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
                                    reduced_data = tsne.fit_transform(embeddings)
                                    title = "t-SNE Visualization of Word Embeddings"
                                
                                # Visualization
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.scatter(reduced_data[:, 0], reduced_data[:, 1], s=100, alpha=0.8)
                                
                                # Add labels to points
                                for i, label in enumerate(labels):
                                    ax.annotate(label, (reduced_data[i, 0] + 0.02, reduced_data[i, 1] + 0.02), 
                                                fontsize=12)
                                
                                ax.set_title(title, fontsize=14)
                                ax.grid(True, alpha=0.3)
                                st.pyplot(fig)
                            elif len(embeddings) == 1:
                                st.info("Need at least two words for visualization. Add more words to see the embedding space.")