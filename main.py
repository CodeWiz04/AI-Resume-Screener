import streamlit as st
import os
import fitz  
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import plotly.graph_objects as go
import time
import nltk

# Import custom styling
from styles import (
    get_custom_css, 
    get_header_html, 
    get_upload_success_html,
    get_score_html,
    get_metric_card_html,
    get_section_header_html,
    get_footer_html
)

# Page config
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar theme toggle
theme = st.sidebar.radio("🎨 Theme Mode", ["Light", "Dark"], index=0)

# Apply custom CSS
st.markdown(get_custom_css(theme), unsafe_allow_html=True)

# Download required NLTK data (run once)
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

download_nltk_data()

# ---------- FILE READING FUNCTIONS ----------
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(file):
    return file.read().decode("utf-8")

# ---------- ULTRA-AGGRESSIVE TEXT PROCESSING ----------
def clean_and_normalize_text(text, aggressive=True):
    """Maximum preservation text cleaning"""
    text = text.lower()
    
    # Preserve ALL technical terms and variations
    replacements = {
        r'c\+\+': 'cplusplus cpp',
        r'c#': 'csharp',
        r'\.net': 'dotnet net',
        r'node\.?js': 'nodejs node javascript',
        r'react\.?js': 'reactjs react javascript',
        r'vue\.?js': 'vuejs vue javascript',
        r'angular\.?js': 'angularjs angular javascript',
        r'express\.?js': 'expressjs express nodejs',
        r'next\.?js': 'nextjs next react',
        r'typescript': 'typescript javascript',
        r'javascript': 'javascript js',
        r'html5': 'html html5',
        r'css3': 'css css3',
        r'mongodb': 'mongodb mongo database',
        r'postgresql': 'postgresql postgres sql database',
        r'mysql': 'mysql sql database',
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Keep alphanumeric and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = ' '.join(text.split())
    
    return text

def extract_all_terms_exhaustive(text):
    """Extract EVERY possible meaningful combination"""
    text_lower = text.lower()
    word_list = text_lower.split()
    
    all_terms = set()
    
    # Single words (1+ chars, very lenient)
    all_terms.update([w for w in word_list if len(w) > 0])
    
    # 2-word phrases
    for i in range(len(word_list) - 1):
        all_terms.add(f"{word_list[i]} {word_list[i+1]}")
    
    # 3-word phrases
    for i in range(len(word_list) - 2):
        all_terms.add(f"{word_list[i]} {word_list[i+1]} {word_list[i+2]}")
    
    # 4-word phrases
    for i in range(len(word_list) - 3):
        all_terms.add(f"{word_list[i]} {word_list[i+1]} {word_list[i+2]} {word_list[i+3]}")
    
    # 5-word phrases (for very specific requirements)
    for i in range(len(word_list) - 4):
        all_terms.add(f"{word_list[i]} {word_list[i+1]} {word_list[i+2]} {word_list[i+3]} {word_list[i+4]}")
    
    return all_terms

def calculate_ultra_keyword_overlap(resume_text, job_text):
    """Maximum possible keyword matching with multiple strategies"""
    
    # Strategy 1: Extract ALL possible terms
    resume_all_terms = extract_all_terms_exhaustive(resume_text)
    job_all_terms = extract_all_terms_exhaustive(job_text)
    
    if not job_all_terms:
        return 85.0, set()  # Very high default
    
    matched_terms = set()
    
    # Level 1: Exact matches
    exact_matches = resume_all_terms.intersection(job_all_terms)
    matched_terms.update(exact_matches)
    
    # Level 2: Substring matches (one contains the other)
    for job_term in job_all_terms:
        if job_term in matched_terms:
            continue
        for resume_term in resume_all_terms:
            if len(job_term) > 1 and len(resume_term) > 1:
                if job_term in resume_term or resume_term in job_term:
                    matched_terms.add(job_term)
                    break
    
    # Level 3: Word overlap in phrases (any common word)


