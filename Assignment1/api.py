from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nltk
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Download necessary NLTK resources
nltk.download('punkt',quiet=True)
nltk.download('averaged_perceptron_tagger_eng',quiet=True)
nltk.download('maxent_ne_chunker_tab',quiet=True)
nltk.download('words',quiet=True)
nltk.download('wordnet',quiet=True)

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    import sys
    print("Downloading spaCy model...")
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

app = FastAPI(
    title="NLP Preprocessing API",
    description="API for text preprocessing functions including tokenization, lemmatization, stemming, POS tagging, NER, and word embeddings",
    version="1.0.0"
)

class TextRequest(BaseModel):
    text: str

class WordRequest(BaseModel):
    word: str
    
class EmbeddingRequest(BaseModel):
    words: List[str]
    num_neighbors: int = 5

# Initialize global variables for embeddings
np.random.seed(42)  # For reproducibility
EMBEDDING_DIMENSION = 50
CORPUS = [
    # Technology
    "Artificial intelligence is transforming technology.",
    "Cloud computing enables business infrastructure.",
    "Blockchain provides secure transaction records.",
    "Internet of Things connects devices.",
    "Quantum computing will revolutionize processing.",
    
    # Science
    "Genetic engineering allows DNA modification.",
    "Renewable energy helps combat climate change.",
    "Space exploration leads to advancements.",
    "Particle physics studies matter constituents.",
    "Neuroscience helps understand brain functions.",
    
    # Arts
    "Renaissance art used realistic techniques.",
    "Abstract art challenges representations.",
    "Literature provides human insights.",
    "Classical music follows harmonic patterns.",
    "Digital art uses technology.",
    
    # Sports
    "Basketball requires coordination and teamwork.",
    "Swimming tests endurance and technique.",
    "Soccer is popular worldwide.",
    "Tennis needs quick reflexes.",
    "Marathon running demands stamina."
]

# Initialize TF-IDF model
def preprocess_text(text):
    """Basic preprocessing: lowercase and remove punctuation"""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Preprocess corpus
PREPROCESSED_CORPUS = [preprocess_text(doc) for doc in CORPUS]

# Create TF-IDF vectorizer
TFIDF_VECTORIZER = TfidfVectorizer(max_features=50)
TFIDF_MATRIX = TFIDF_VECTORIZER.fit_transform(PREPROCESSED_CORPUS)
FEATURE_NAMES = TFIDF_VECTORIZER.get_feature_names_out()

# Create word embeddings dictionary
def create_mock_glove_embeddings():
    """Create mock GloVe embeddings"""
    all_words = set()
    for doc in PREPROCESSED_CORPUS:
        all_words.update(doc.split())
    
    # Create mock embeddings dictionary
    glove_dict = {}
    for word in all_words:
        glove_dict[word] = np.random.rand(EMBEDDING_DIMENSION)
    
    # Add some common words
    common_words = ['the', 'and', 'of', 'to', 'in', 'is', 'that', 'for', 'technology', 'science', 'art', 'sport']
    for word in common_words:
        if word not in glove_dict:
            glove_dict[word] = np.random.rand(EMBEDDING_DIMENSION)
    
    return glove_dict

# Initialize embedding dictionary
WORD_EMBEDDINGS = create_mock_glove_embeddings()

def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors"""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

def find_similar_words(word, word_dict, n=5):
    """Find most similar words to a given word"""
    if word not in word_dict:
        return []
        
    word_vector = word_dict[word]
    similarities = []
    
    for w, vec in word_dict.items():
        if w != word:  # Skip the query word itself
            sim = cosine_similarity(word_vector, vec)
            similarities.append((w, float(sim)))  # Convert numpy float to Python float for JSON serialization
            
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:n]

def get_wordnet_pos(tag):
    """Map POS tag to first character used by WordNetLemmatizer"""
    if tag.startswith('J'):
        return 'a'  # Adjective
    elif tag.startswith('V'):
        return 'v'  # Verb
    elif tag.startswith('N'):
        return 'n'  # Noun
    elif tag.startswith('R'):
        return 'r'  # Adverb
    else:
        return 'n'  # Default to noun

@app.post("/tokenize")
def tokenize(request: TextRequest):
    """Tokenizes input text into sentences and words"""
    text = request.text
    
    # NLTK tokenization
    nltk_sentences = sent_tokenize(text)
    nltk_words = word_tokenize(text)
    
    # spaCy tokenization
    doc = nlp(text)
    spacy_sentences = [sent.text for sent in doc.sents]
    spacy_tokens = [token.text for token in doc]
    
    return {
        "nltk": {
            "sentences": nltk_sentences,
            "words": nltk_words
        },
        "spacy": {
            "sentences": spacy_sentences,
            "tokens": spacy_tokens
        }
    }

@app.post("/lemmatize")
def lemmatize(request: TextRequest):
    """Lemmatizes input text using NLTK and spaCy"""
    text = request.text
    
    # NLTK lemmatization
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    words_with_pos = pos_tag(words)
    lemmas_nltk = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in words_with_pos]
    
    # spaCy lemmatization
    doc = nlp(text)
    lemmas_spacy = [token.lemma_ for token in doc]
    
    return {
        "original": words,
        "nltk_lemmas": lemmas_nltk,
        "spacy_lemmas": lemmas_spacy,
        "nltk_pairs": [{"original": w, "lemma": l} for w, l in zip(words, lemmas_nltk) if w != l],
        "spacy_pairs": [{"original": t.text, "lemma": t.lemma_} for t in doc if t.text != t.lemma_]
    }

@app.post("/stem")
def stem(request: TextRequest):
    """Applies stemming to input text using Porter, Lancaster, and Snowball stemmers"""
    text = request.text
    
    # Initialize stemmers
    porter_stemmer = PorterStemmer()
    lancaster_stemmer = LancasterStemmer()
    snowball_stemmer = SnowballStemmer('english')
    
    # Tokenize and stem
    words = word_tokenize(text)
    stems_porter = [porter_stemmer.stem(word) for word in words]
    stems_lancaster = [lancaster_stemmer.stem(word) for word in words]
    stems_snowball = [snowball_stemmer.stem(word) for word in words]
    
    return {
        "original": words,
        "porter_stems": stems_porter,
        "lancaster_stems": stems_lancaster,
        "snowball_stems": stems_snowball,
        "comparison": [
            {
                "original": w,
                "porter": p,
                "lancaster": l,
                "snowball": s
            } 
            for w, p, l, s in zip(words, stems_porter, stems_lancaster, stems_snowball)
        ]
    }

@app.post("/pos-tag")
def pos_tagging(request: TextRequest):
    """Performs Part-of-Speech tagging on input text"""
    text = request.text
    
    # NLTK POS tagging
    words = word_tokenize(text)
    nltk_pos = pos_tag(words)
    
    # spaCy POS tagging
    doc = nlp(text)
    spacy_pos = [{"text": token.text, "pos": token.pos_, "tag": token.tag_, 
                  "explanation": spacy.explain(token.tag_)} for token in doc]
    
    return {
        "nltk": [{"text": word, "pos": tag} for word, tag in nltk_pos],
        "spacy": spacy_pos
    }

@app.post("/ner")
def named_entity_recognition(request: TextRequest):
    """Performs Named Entity Recognition on input text"""
    text = request.text
    
    # NLTK NER
    words = word_tokenize(text)
    nltk_pos = pos_tag(words)
    nltk_ner = ne_chunk(nltk_pos)
    
    # Extract named entities from NLTK
    named_entities_nltk = []
    for chunk in nltk_ner:
        if hasattr(chunk, 'label'):
            entity = ' '.join(c[0] for c in chunk)
            entity_type = chunk.label()
            named_entities_nltk.append({"text": entity, "type": entity_type})
    
    # spaCy NER
    doc = nlp(text)
    named_entities_spacy = [
        {
            "text": ent.text,
            "type": ent.label_,
            "explanation": spacy.explain(ent.label_),
            "start": ent.start_char,
            "end": ent.end_char
        } for ent in doc.ents
    ]
    
    return {
        "nltk": named_entities_nltk,
        "spacy": named_entities_spacy
    }

@app.post("/process-all")
def process_all(request: TextRequest):
    """Performs all preprocessing functions on the input text"""
    return {
        "tokenization": tokenize(request),
        "lemmatization": lemmatize(request),
        "stemming": stem(request),
        "pos_tagging": pos_tagging(request),
        "ner": named_entity_recognition(request)
    }

@app.post("/get-word-embedding")
def get_word_embedding(request: WordRequest):
    """Returns the embedding for a single word"""
    word = preprocess_text(request.word)
    
    # Check if word is in vocabulary
    if word not in WORD_EMBEDDINGS:
        raise HTTPException(status_code=404, detail=f"Word '{word}' not found in vocabulary")
    
    # Get embedding
    embedding = WORD_EMBEDDINGS[word].tolist()
    return {
        "word": word,
        "embedding": embedding,
        "dimension": EMBEDDING_DIMENSION
    }

@app.post("/get-similar-words")
def get_similar_words(request: WordRequest):
    """Returns words most similar to the input word"""
    word = preprocess_text(request.word)
    
    # Check if word is in vocabulary
    if word not in WORD_EMBEDDINGS:
        raise HTTPException(status_code=404, detail=f"Word '{word}' not found in vocabulary")
    
    # Find similar words
    similar_words = find_similar_words(word, WORD_EMBEDDINGS, n=10)
    return {
        "word": word,
        "similar_words": [{"word": w, "similarity": s} for w, s in similar_words]
    }

@app.post("/get-multiple-embeddings")
def get_multiple_embeddings(request: EmbeddingRequest):
    """Returns embeddings and similar words for multiple input words"""
    results = []
    
    for word in request.words:
        processed_word = preprocess_text(word)
        
        # Check if word is in vocabulary
        if processed_word in WORD_EMBEDDINGS:
            embedding = WORD_EMBEDDINGS[processed_word].tolist()
            similar_words = find_similar_words(processed_word, WORD_EMBEDDINGS, n=request.num_neighbors)
            
            results.append({
                "word": processed_word,
                "in_vocabulary": True,
                "embedding": embedding,
                "similar_words": [{"word": w, "similarity": s} for w, s in similar_words]
            })
        else:
            results.append({
                "word": processed_word,
                "in_vocabulary": False
            })
    
    return {
        "results": results,
        "dimension": EMBEDDING_DIMENSION
    }

@app.get("/vocabulary")
def get_vocabulary():
    """Returns available words in the embedding vocabulary"""
    vocab_words = list(WORD_EMBEDDINGS.keys())
    return {
        "vocabulary_size": len(vocab_words),
        "words": vocab_words[:100]  # Limit to 100 words for API response
    }

@app.get("/")
def root():
    """Root endpoint providing API information"""
    return {
        "message": "NLP Preprocessing and Word Embedding API",
        "preprocessing_endpoints": ["/tokenize", "/lemmatize", "/stem", "/pos-tag", "/ner", "/process-all"],
        "embedding_endpoints": ["/get-word-embedding", "/get-similar-words", "/get-multiple-embeddings", "/vocabulary"],
        "usage": "Send a POST request with JSON payload: {'text': 'Your text here'} or {'word': 'your_word'}"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)