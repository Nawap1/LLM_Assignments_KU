from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nltk
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from typing import List, Dict, Any

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
    description="API for text preprocessing functions including tokenization, lemmatization, stemming, POS tagging, and NER",
    version="1.0.0"
)

class TextRequest(BaseModel):
    text: str

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

@app.get("/")
def root():
    """Root endpoint providing API information"""
    return {
        "message": "NLP Preprocessing API",
        "endpoints": ["/tokenize", "/lemmatize", "/stem", "/pos-tag", "/ner", "/process-all"],
        "usage": "Send a POST request with JSON payload: {'text': 'Your text here'}"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)