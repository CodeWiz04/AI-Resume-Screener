import streamlit as st
import os
import fitz  # PyMuPDF for PDF
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

# ---------- ENHANCED TEXT PROCESSING ----------
def clean_and_normalize_text(text):
    """Enhanced text cleaning and normalization"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = ' '.join(text.split())
    return text

def extract_keywords(text, max_features=100):
    """Extract important keywords using TF-IDF"""
    try:
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        
        tokens = word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words and len(token) > 2]
        
        processed_text = ' '.join(tokens)
        
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([processed_text])
        
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        
        keyword_scores = list(zip(feature_names, tfidf_scores))
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [kw[0] for kw in keyword_scores[:max_features]]
    except:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return list(set(words))

def calculate_keyword_overlap(resume_keywords, job_keywords):
    """Calculate keyword overlap percentage"""
    resume_set = set(resume_keywords)
    job_set = set(job_keywords)
    
    if not job_set:
        return 0, set()
    
    overlap = resume_set.intersection(job_set)
    overlap_percentage = (len(overlap) / len(job_set)) * 100
    
    return overlap_percentage, overlap

def chunk_text(text, chunk_size=500):
    """Split text into smaller chunks for better semantic analysis"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer("all-MiniLM-L6-v2")

def calculate_semantic_similarity(resume_text, job_desc):
    """Calculate semantic similarity using sentence transformers"""
    model = load_sentence_transformer()
    
    resume_chunks = chunk_text(resume_text, 200)
    
    resume_embeddings = model.encode(resume_chunks)
    job_embedding = model.encode([job_desc])
    
    similarities = cosine_similarity(resume_embeddings, job_embedding)
    max_similarity = np.max(similarities)
    avg_similarity = np.mean(similarities)
    
    return max_similarity, avg_similarity

def calculate_tfidf_similarity(resume_text, job_desc):
    """Calculate TF-IDF based similarity"""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
    
    try:
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    except:
        return 0

def calculate_composite_score(semantic_max, semantic_avg, tfidf_sim, keyword_overlap):
    """Calculate a composite matching score"""
    composite_score = (
        semantic_max * 0.3 +
        semantic_avg * 0.2 +
        tfidf_sim * 0.3 +
        (keyword_overlap / 100) * 0.2
    )
    return composite_score * 100

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

# Sidebar with enhanced styling
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

# Main content area with perfectly aligned boxes
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown(get_section_header_html("📄", "Upload Resume"), unsafe_allow_html=True)
    fileName = st.file_uploader(
        "Choose your resume file",
        type=["pdf", "docx", "txt"],
        help="Supported formats: PDF, DOCX, TXT",
        label_visibility="collapsed"
    )
    
    # Show upload status below the upload area
    if fileName is not None:
        st.markdown(
            get_upload_success_html(fileName.name, fileName.size),
            unsafe_allow_html=True
        )

with col2:
    st.markdown(get_section_header_html("💼", "Job Description"), unsafe_allow_html=True)
    job_desc = st.text_area(
        "Paste the job description here:",
        height=240,
        placeholder="Paste the complete job description including:\n\n• Required qualifications and skills\n• Job responsibilities\n• Experience requirements\n• Technical competencies\n• Educational background\n\nThe more detailed, the better the analysis!",
        help="Include the complete job posting for better analysis",
        label_visibility="collapsed"
    )

# Analysis button with custom styling
st.markdown("<br>", unsafe_allow_html=True)
col_center = st.columns([1, 2, 1])[1]
with col_center:
    analyze_button = st.button(
        "🚀 Analyze Resume Match",
        type="primary",
        use_container_width=True,
        help="Click to start comprehensive resume analysis"
    )

if analyze_button:
    if fileName is None or not job_desc.strip():
        st.error("⚠ Please upload a resume and enter a job description.")
    else:
        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("🔍 Analyzing resume and job description..."):
            # Update progress
            status_text.text("📄 Extracting text from resume...")
            progress_bar.progress(20)
            
            # Extract text
            _, ext = os.path.splitext(fileName.name)
            ext = ext.lower()

            if ext == ".pdf":
                resume_text = extract_text_from_pdf(fileName)
            elif ext == ".docx":
                resume_text = extract_text_from_docx(fileName)
            elif ext == ".txt":
                resume_text = extract_text_from_txt(fileName)
            else:
                st.error("❌ Unsupported file type.")
                st.stop()

            # Clean texts
            status_text.text("🧹 Cleaning and preprocessing text...")
            progress_bar.progress(40)
            
            resume_clean = clean_and_normalize_text(resume_text)
            job_clean = clean_and_normalize_text(job_desc)
            
            if not resume_clean or not job_clean:
                st.error("❌ Could not extract text from the uploaded file or job description is empty.")
                st.stop()

    # Perform analysis
            status_text.text("🤖 Running AI analysis...")
            progress_bar.progress(60)
            
            # Initialize variables
            semantic_max = semantic_avg = tfidf_similarity = keyword_overlap_pct = 0
            overlapping_keywords = set()
            
            if analysis_type in ["Composite Score (Recommended)", "Semantic Only"]:
                semantic_max, semantic_avg = calculate_semantic_similarity(resume_clean, job_clean)
                
            if analysis_type in ["Composite Score (Recommended)", "TF-IDF Only"]:
                tfidf_similarity = calculate_tfidf_similarity(resume_clean, job_clean)
                
            if analysis_type in ["Composite Score (Recommended)", "Keyword Only"]:
                resume_keywords = extract_keywords(resume_clean)
                job_keywords = extract_keywords(job_clean)
                keyword_overlap_pct, overlapping_keywords = calculate_keyword_overlap(resume_keywords, job_keywords)

            status_text.text("📊 Calculating final scores...")
            progress_bar.progress(80)

        
            if analysis_type == "Composite Score (Recommended)":
                final_score = calculate_composite_score(semantic_max, semantic_avg, tfidf_similarity, keyword_overlap_pct)
                score_type = "Composite"
            elif analysis_type == "Semantic Only":
                final_score = semantic_max * 100
                score_type = "Semantic"
            elif analysis_type == "Keyword Only":
                final_score = keyword_overlap_pct
                score_type = "Keyword Overlap"
            elif analysis_type == "TF-IDF Only":
                final_score = tfidf_similarity * 100
                score_type = "TF-IDF"

            final_score = min(round(final_score, 2), 100)
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            time.sleep(0.5)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

        # Display results with custom styling
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Main score display
        score_html, message = get_score_html(final_score, score_type)
        st.markdown(score_html, unsafe_allow_html=True)
        
        st.info(f"📝 *Analysis Result:* {message}")

        # Detailed analysis section
        if show_details:
            st.markdown("---")
            st.markdown("## 📊 Detailed Analysis")
            
            # Create metrics in a beautiful layout
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.markdown(
                    get_metric_card_html(f"{semantic_max*100:.1f}%", "Best Semantic Match"),
                    unsafe_allow_html=True
                )
            
            with metric_col2:
                st.markdown(
                    get_metric_card_html(f"{semantic_avg*100:.1f}%", "Avg Semantic Match"),
                    unsafe_allow_html=True
                )
            
            with metric_col3:
                st.markdown(
                    get_metric_card_html(f"{tfidf_similarity*100:.1f}%", "TF-IDF Similarity"),
                    unsafe_allow_html=True
                )
            
            with metric_col4:
                st.markdown(
                    get_metric_card_html(f"{keyword_overlap_pct:.1f}%", "Keyword Overlap"),
                    unsafe_allow_html=True
                )

            # Score visualization
            if show_visualization and analysis_type == "Composite Score (Recommended)":
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📈 Score Breakdown")
                scores_dict = {
                    "Semantic Max": semantic_max * 100,
                    "Semantic Avg": semantic_avg * 100,
                    "TF-IDF": tfidf_similarity * 100,
                    "Keywords": keyword_overlap_pct
                }
                fig = create_score_visualization(scores_dict)
                st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(get_footer_html(), unsafe_allow_html=True)