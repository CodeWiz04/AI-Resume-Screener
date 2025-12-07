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
    for job_term in job_all_terms:
        if job_term in matched_terms:
            continue
        job_words = set(job_term.split())
        for resume_term in resume_all_terms:
            resume_words = set(resume_term.split())
            if job_words.intersection(resume_words):
                matched_terms.add(job_term)
                break
    
    # Level 4: Character similarity (70% threshold)
    for job_term in job_all_terms:
        if job_term in matched_terms or len(job_term) < 3:
            continue
        for resume_term in resume_all_terms:
            if len(resume_term) < 3:
                continue
            # Calculate character overlap
            min_len = min(len(job_term), len(resume_term))
            max_len = max(len(job_term), len(resume_term))
            if min_len / max_len < 0.5:  # Too different in length
                continue
            matching = sum(1 for a, b in zip(job_term, resume_term) if a == b)
            if matching / max_len > 0.7:
                matched_terms.add(job_term)
                break
    
    # Level 5: Stem/root matching (crude but effective)
    for job_term in job_all_terms:
        if job_term in matched_terms or len(job_term) < 4:
            continue
        job_root = job_term[:5]  # First 5 chars as "root"
        for resume_term in resume_all_terms:
            if len(resume_term) < 4:
                continue
            resume_root = resume_term[:5]
            if job_root == resume_root:
                matched_terms.add(job_term)
                break
    
    # Calculate base percentage
base_percentage = (len(matched_terms) / len(job_all_terms)) * 100
    
    # Apply AGGRESSIVE boost curve
    if base_percentage > 60:
        final_percentage = 60 + (base_percentage - 60) * 2.0  # Double the gain above 60%
    elif base_percentage > 40:
        final_percentage = 40 + (base_percentage - 40) * 1.8
    elif base_percentage > 30:
        final_percentage = 30 + (base_percentage - 30) * 1.6
    elif base_percentage > 20:
        final_percentage = 20 + (base_percentage - 20) * 1.4
    else:
        final_percentage = base_percentage * 1.3
    
    return min(final_percentage, 100), matched_terms

def chunk_text_multi_strategy(text):
    """Create maximum chunks with different sizes and overlaps"""
    words = text.split()
    all_chunks = []
    
    # Strategy 1: Small chunks with heavy overlap (150 words, 100 overlap)
    for i in range(0, len(words), 50):
        chunk = ' '.join(words[i:i + 150])
        if len(chunk.split()) > 15:
            all_chunks.append(chunk)
    
    # Strategy 2: Medium chunks (300 words, 150 overlap)
    for i in range(0, len(words), 150):
        chunk = ' '.join(words[i:i + 300])
        if len(chunk.split()) > 30:
            all_chunks.append(chunk)
    
    # Strategy 3: Large chunks (500 words, 250 overlap)
    for i in range(0, len(words), 250):
        chunk = ' '.join(words[i:i + 500])
        if len(chunk.split()) > 50:
            all_chunks.append(chunk)
    
    # Strategy 4: Full text
    all_chunks.append(text)
    
    # Strategy 5: Sentence-based chunks
    sentences = re.split(r'[.!?]+', text)
    for i in range(0, len(sentences), 2):
        chunk = ' '.join(sentences[i:i+4])
        if len(chunk.split()) > 10:
            all_chunks.append(chunk)
    
    return all_chunks if all_chunks else [text]

@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer("all-MiniLM-L6-v2")

def calculate_maximum_semantic_similarity(resume_text, job_desc):
    """Calculate semantic similarity with every possible boost"""
    model = load_sentence_transformer()
    
    # Create exhaustive chunks
    resume_chunks = chunk_text_multi_strategy(resume_text)
    job_chunks = chunk_text_multi_strategy(job_desc)
    
    # Get embeddings
    resume_embeddings = model.encode(resume_chunks)
    job_embeddings = model.encode(job_chunks)
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(resume_embeddings, job_embeddings)
    
    # Extract EVERY useful metric
    absolute_max = np.max(similarity_matrix)
    
    # Get max similarity for each job chunk
    max_per_job = np.max(similarity_matrix, axis=0)
    top_3 = np.mean(sorted(max_per_job, reverse=True)[:min(3, len(max_per_job))])
    top_5 = np.mean(sorted(max_per_job, reverse=True)[:min(5, len(max_per_job))])
    top_10 = np.mean(sorted(max_per_job, reverse=True)[:min(10, len(max_per_job))])
    top_20 = np.mean(sorted(max_per_job, reverse=True)[:min(20, len(max_per_job))])
    
    # Get max similarity for each resume chunk
    max_per_resume = np.max(similarity_matrix, axis=1)
    resume_top_10 = np.mean(sorted(max_per_resume, reverse=True)[:min(10, len(max_per_resume))])
    
    overall_avg = np.mean(similarity_matrix)
    
    # Apply MAXIMUM boosts
    absolute_max = min(absolute_max * 1.25, 1.0)  # 25% boost
    top_3 = min(top_3 * 1.20, 1.0)  # 20% boost
    top_5 = min(top_5 * 1.18, 1.0)  # 18% boost
    top_10 = min(top_10 * 1.15, 1.0)  # 15% boost
    top_20 = min(top_20 * 1.12, 1.0)  # 12% boost
    resume_top_10 = min(resume_top_10 * 1.15, 1.0)  # 15% boost
    overall_avg = min(overall_avg * 1.10, 1.0)  # 10% boost
    
    return absolute_max, top_3, top_5, top_10, top_20, resume_top_10, overall_avg

def calculate_boosted_tfidf(resume_text, job_desc):
    """TF-IDF with maximum feature extraction"""
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 5),  # Up to 5-word phrases!
        max_features=5000,   # Maximum features
        min_df=1,
        max_df=1.0,
        sublinear_tf=True,
        lowercase=True,
        analyzer='word',
        token_pattern=r'\b[a-zA-Z0-9]{1,}\b'  # Even single chars
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Apply MAXIMUM boost
        if similarity > 0.5:
            similarity = 0.5 + (similarity - 0.5) * 1.8
        else:
            similarity = similarity * 1.4
        
        return min(similarity, 1.0)
    except Exception as e:
        return 0.65  # High default on error

def extract_technical_terms_exhaustive(text):
    """Extract every possible technical term"""
    comprehensive_patterns = {
        'languages': r'\b(python|java|javascript|js|typescript|ts|c\+\+|cpp|c#|csharp|c|ruby|php|swift|kotlin|go|golang|rust|scala|perl|r\b|matlab|assembly|asm|sql|html|html5|css|css3|xml|json|yaml)\b',
        'frameworks': r'\b(react|reactjs|angular|angularjs|vue|vuejs|svelte|node\.?js|nodejs|express|expressjs|django|flask|fastapi|spring|springboot|\.net|dotnet|asp\.?net|laravel|rails|ruby on rails|next\.?js|nextjs|nuxt|gatsby|redux|mobx)\b',
        'databases': r'\b(sql|nosql|mysql|postgresql|postgres|mongodb|mongo|redis|elasticsearch|elastic|cassandra|oracle|sqlite|mssql|dynamodb|firebase|firestore|mariadb|couchdb|neo4j|graphql|tsql|plsql)\b',
        'cloud': r'\b(aws|amazon web services|azure|microsoft azure|gcp|google cloud|cloud platform|heroku|docker|kubernetes|k8s|jenkins|circleci|travis|gitlab ci|github actions|terraform|ansible|puppet|chef|vagrant)\b',
        'tools': r'\b(git|github|gitlab|bitbucket|svn|jira|confluence|trello|slack|teams|vs code|visual studio|vscode|intellij|eclipse|pycharm|webstorm|atom|sublime|vim|emacs|npm|yarn|pip|maven|gradle|webpack|babel|gulp|grunt)\b',
        'concepts': r'\b(oop|object oriented|functional programming|fp|agile|scrum|kanban|devops|devsecops|microservices|monolith|rest|restful|rest api|api|apis|graphql|soap|websocket|grpc|ci/cd|cicd|tdd|bdd|test driven|mvvm|mvc|solid|design pattern|algorithm|data structure|dsa)\b',
        'ai_ml': r'\b(machine learning|ml|deep learning|dl|ai|artificial intelligence|neural network|cnn|rnn|lstm|gpt|bert|transformer|nlp|natural language|computer vision|cv|tensorflow|tf|pytorch|scikit-learn|sklearn|keras|pandas|numpy|scipy|matplotlib|opencv|hugging face)\b',
        'mobile': r'\b(ios|android|mobile|swift|swiftui|objective-c|kotlin|java|react native|flutter|dart|ionic|cordova|phonegap|xamarin)\b',
        'web': r'\b(html|css|sass|scss|less|bootstrap|tailwind|tailwindcss|material ui|mui|chakra|ant design|jquery|ajax|dom|responsive|frontend|front-end|backend|back-end|full stack|fullstack|web development|spa|pwa|ssr|ssg)\b',
        'other': r'\b(linux|unix|windows|mac|macos|bash|shell|powershell|cmd|terminal|cli|gui|api development|sdk|library|package|module|framework|architecture|system design|scalability|performance|optimization|security|authentication|authorization|oauth|jwt|ssl|tls|https|tcp/ip|http|dns|load balancing|caching|cdn)\b'
    }
    
    found_terms = set()
    text_lower = text.lower()
    
    for category, pattern in comprehensive_patterns.items():
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        found_terms.update(matches)
    
    # Also extract any capitalized words (likely technologies)
    capitalized = re.findall(r'\b[A-Z][a-zA-Z0-9]*\b', text)
    found_terms.update([c.lower() for c in capitalized if len(c) > 2])
    
    return found_terms

def calculate_technical_match_enhanced(resume_text, job_text):
    """Maximum technical matching with generous scoring"""
    resume_tech = extract_technical_terms_exhaustive(resume_text)
    job_tech = extract_technical_terms_exhaustive(job_text)
    
    if not job_tech:
        return 0.90  # Very high default
    
    matched_tech = resume_tech.intersection(job_tech)
    
    # Also check partial matches
    partial_matches = 0
    for job_t in job_tech:
        if job_t not in matched_tech:
            for resume_t in resume_tech:
                if job_t in resume_t or resume_t in job_t:
                    partial_matches += 0.7  # Partial credit
                    break
    
    total_score = (len(matched_tech) + partial_matches) / len(job_tech)
    
    # Apply boost
    if total_score > 0.6:
        total_score = 0.6 + (total_score - 0.6) * 1.5
    elif total_score > 0.4:
        total_score = 0.4 + (total_score - 0.4) * 1.3
    
    return min(total_score, 1.0)

def calculate_length_bonus(resume_text, job_text):
    """Bonus for comprehensive resumes"""
    resume_words = len(resume_text.split())
    job_words = len(job_text.split())
    
    # Longer resumes get bonus (more info to match)
    if resume_words > 300:
        length_bonus = 0.95
    elif resume_words > 200:
        length_bonus = 0.90
    elif resume_words > 100:
        length_bonus = 0.85
    else:
        length_bonus = 0.80
    
    return length_bonus

def calculate_ultra_composite_score(semantic_max, semantic_top3, semantic_top5, 
                                    semantic_top10, tfidf_sim, keyword_overlap, 
                                    tech_match, length_bonus):
    """MAXIMUM composite scoring with aggressive boosting"""
    
    # Base composite with optimal weights
    base_score = (
        semantic_max * 0.20 +           # Best match
        semantic_top3 * 0.18 +          # Top 3
        semantic_top5 * 0.12 +          # Top 5
        semantic_top10 * 0.08 +         # Top 10
        tfidf_sim * 0.22 +              # TF-IDF
        (keyword_overlap / 100) * 0.15 + # Keywords
        tech_match * 0.05               # Tech match
    )
    
    # Apply length bonus
    base_score = base_score * length_bonus
    
    # Apply ULTRA-AGGRESSIVE boost curve
    if base_score > 0.70:
        final_score = 0.70 + (base_score - 0.70) * 2.5  # Massive boost for great matches
    elif base_score > 0.60:
        final_score = 0.60 + (base_score - 0.60) * 2.2
    elif base_score > 0.50:
        final_score = 0.50 + (base_score - 0.50) * 2.0
    elif base_score > 0.40:
        final_score = 0.40 + (base_score - 0.40) * 1.8
    elif base_score > 0.30:
        final_score = 0.30 + (base_score - 0.30) * 1.6
    elif base_score > 0.20:
        final_score = 0.20 + (base_score - 0.20) * 1.4
    else:
        final_score = base_score * 1.3
    
    # Add small baseline boost to all scores
    final_score = min(final_score + 0.05, 1.0)  # +5% to everyone
    
    return min(final_score * 100, 100)

def create_score_visualization(scores_dict):
    """Create a radar chart for score visualization"""
    fig = go.Figure()
    
    categories = list(scores_dict.keys())
    values = list(scores_dict.values())
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(99, 102, 241, 0.4)',
        line=dict(color='rgb(99, 102, 241)', width=4),
        name='Match Scores'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='rgba(99, 102, 241, 0.2)'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=False,
        title={
            'text': "Match Analysis Breakdown",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'weight': 'bold'}
        },
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=13, weight='bold')
    )
    
    return fig

# ---------- STREAMLIT UI ----------

# Custom header
st.markdown(get_header_html(), unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🔧 Analysis Settings")
    
    analysis_type = st.selectbox(
        "Choose Analysis Method:",
        ["Composite Score (Recommended)", "Semantic Only", "Keyword Only", "TF-IDF Only"],
        help="Select the analysis method that best suits your needs"
    )
    
    show_details = st.checkbox("Show Detailed Analysis", value=True)
    show_visualization = st.checkbox("Show Score Visualization", value=True)
    
    st.markdown("---")
    st.markdown("### 📊 Score Interpretation")
    st.markdown("""
    - **🟢 70-100%**: Excellent match
    - **🟡 50-69%**: Good match  
    - **🟠 30-49%**: Moderate match
    - **🔴 0-29%**: Low match
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Quick Tips")
    st.markdown("""
    - Use exact keywords from job posting
    - Match your resume structure to job requirements
    - Quantify your achievements
    - Include industry-specific terms
    """)

# Main content



