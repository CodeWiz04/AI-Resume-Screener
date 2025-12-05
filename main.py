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
    """Enhanced text cleaning with better preservation of important terms"""
    # Convert to lowercase but preserve important patterns
    text = text.lower()
    
    # Preserve specific patterns like C++, .NET, etc.
    text = re.sub(r'c\+\+', 'cplusplus', text)
    text = re.sub(r'c#', 'csharp', text)
    text = re.sub(r'\.net', 'dotnet', text)
    text = re.sub(r'node\.js', 'nodejs', text)
    
    # Remove special characters but keep alphanumeric
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def extract_keywords(text, max_features=150):
    """Extract important keywords using improved TF-IDF with better parameters"""
    try:
        # Custom stop words - remove common words but keep technical terms
        stop_words = set(stopwords.words('english'))
        # Remove some stop words that might be important in job context
        technical_words = {'experience', 'using', 'including', 'required', 'preferred'}
        stop_words = stop_words - technical_words
        
        lemmatizer = WordNetLemmatizer()
        
        tokens = word_tokenize(text.lower())
        # Keep words that are alphanumeric and either not in stop words or longer than 2 chars
        tokens = [
            lemmatizer.lemmatize(token) 
            for token in tokens 
            if (token.isalpha() or token.isalnum()) and len(token) > 1
        ]
        
        processed_text = ' '.join(tokens)
        
        # Improved TF-IDF with better parameters
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 3),  # Include trigrams for better phrase matching
            min_df=1,
            max_df=0.95,
            sublinear_tf=True
        )
        
        tfidf_matrix = vectorizer.fit_transform([processed_text])
        
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        
        keyword_scores = list(zip(feature_names, tfidf_scores))
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [kw[0] for kw in keyword_scores[:max_features]]
    except Exception as e:
        # Fallback to simple word extraction
        words = re.findall(r'\b[a-zA-Z0-9]{2,}\b', text.lower())
        return list(set(words))[:max_features]

def calculate_keyword_overlap(resume_keywords, job_keywords):
    """Calculate keyword overlap with fuzzy matching for better results"""
    resume_set = set(resume_keywords)
    job_set = set(job_keywords)
    
    if not job_set:
        return 0, set()
    
    # Exact match
    exact_overlap = resume_set.intersection(job_set)
    
    # Fuzzy match - check if resume keywords contain job keywords or vice versa
    fuzzy_overlap = set()
    for job_kw in job_set:
        for resume_kw in resume_set:
            # Check substring matches (e.g., "javascript" matches "java")
            if len(job_kw) > 3 and len(resume_kw) > 3:
                if job_kw in resume_kw or resume_kw in job_kw:
                    fuzzy_overlap.add(job_kw)
                    break
    
    # Combine exact and fuzzy matches
    total_overlap = exact_overlap.union(fuzzy_overlap)
    
    # Calculate percentage with better weighting
    overlap_percentage = (len(total_overlap) / len(job_set)) * 100
    
    return overlap_percentage, total_overlap

def chunk_text(text, chunk_size=300, overlap=50):
    """Split text into overlapping chunks for better semantic analysis"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.split()) > 20:  # Only add chunks with substantial content
            chunks.append(chunk)
    
    return chunks if chunks else [text]

@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer("all-MiniLM-L6-v2")

def calculate_semantic_similarity(resume_text, job_desc):
    """Calculate semantic similarity with improved chunking strategy"""
    model = load_sentence_transformer()
    
    # Create chunks with overlap
    resume_chunks = chunk_text(resume_text, chunk_size=300, overlap=50)
    job_chunks = chunk_text(job_desc, chunk_size=300, overlap=50)
    
    # Get embeddings
    resume_embeddings = model.encode(resume_chunks)
    job_embeddings = model.encode(job_chunks)
    
    # Calculate similarity matrix between all resume and job chunks
    similarity_matrix = cosine_similarity(resume_embeddings, job_embeddings)
    
    # Get max similarity for each job chunk (best match)
    max_similarities = np.max(similarity_matrix, axis=0)
    
    # Calculate metrics
    max_similarity = np.max(max_similarities)
    avg_similarity = np.mean(max_similarities)
    top_3_avg = np.mean(sorted(max_similarities, reverse=True)[:min(3, len(max_similarities))])
    
    return max_similarity, avg_similarity, top_3_avg

def calculate_tfidf_similarity(resume_text, job_desc):
    """Calculate TF-IDF based similarity with improved parameters"""
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),  # Include trigrams
        max_features=2000,
        min_df=1,
        sublinear_tf=True
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    except Exception as e:
        return 0

def extract_skills_and_technologies(text):
    """Extract common technical skills and technologies"""
    # Common programming languages and technologies
    tech_patterns = [
        r'\b(python|java|javascript|typescript|c\+\+|c#|ruby|php|swift|kotlin|go|rust)\b',
        r'\b(react|angular|vue|node\.?js|express|django|flask|spring|\.net)\b',
        r'\b(sql|mysql|postgresql|mongodb|redis|elasticsearch|oracle)\b',
        r'\b(aws|azure|gcp|docker|kubernetes|jenkins|git|ci/cd)\b',
        r'\b(html|css|sass|less|bootstrap|tailwind)\b',
        r'\b(machine learning|deep learning|ai|nlp|computer vision|data science)\b',
        r'\b(agile|scrum|devops|microservices|rest|api|graphql)\b',
        r'\b(tensorflow|pytorch|scikit-learn|pandas|numpy)\b',
        r'\b(assembly|oop|algorithms|data structures|system programming)\b',
    ]
    
    found_skills = set()
    text_lower = text.lower()
    
    for pattern in tech_patterns:
        matches = re.findall(pattern, text_lower)
        found_skills.update(matches)
    
    return found_skills

def calculate_skills_match(resume_text, job_text):
    """Calculate how many required skills are present in resume"""
    resume_skills = extract_skills_and_technologies(resume_text)
    job_skills = extract_skills_and_technologies(job_text)
    
    if not job_skills:
        return 0.5  # Neutral score if no skills detected
    
    matched_skills = resume_skills.intersection(job_skills)
    match_percentage = len(matched_skills) / len(job_skills)
    
    return match_percentage

def calculate_composite_score(semantic_max, semantic_avg, top_3_avg, tfidf_sim, keyword_overlap, skills_match):
    """Calculate composite score with improved weighting"""
    # Improved weighting that rewards strong matches
    composite_score = (
        semantic_max * 0.25 +      # Best semantic match
        top_3_avg * 0.20 +          # Top matches average
        semantic_avg * 0.10 +       # Overall semantic average
        tfidf_sim * 0.25 +          # TF-IDF similarity
        (keyword_overlap / 100) * 0.15 +  # Keyword overlap
        skills_match * 0.05         # Skills match bonus
    )
    
    # Apply scaling to make scores more realistic (avoiding too low scores)
    # If basic match exists, ensure minimum reasonable score
    if composite_score > 0.3:
        composite_score = 0.3 + (composite_score - 0.3) * 1.4
    
    return min(composite_score * 100, 100)

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
        st.error("⚠️ Please upload a resume and enter a job description.")
    else:
        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("🔍 Analyzing resume and job description..."):
            # Update progress
            status_text.text("📄 Extracting text from resume...")
            progress_bar.progress(15)
            
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
            progress_bar.progress(30)
            
            resume_clean = clean_and_normalize_text(resume_text)
            job_clean = clean_and_normalize_text(job_desc)
            
            if not resume_clean or not job_clean:
                st.error("❌ Could not extract text from the uploaded file or job description is empty.")
                st.stop()

            # Perform analysis
            status_text.text("🤖 Running AI analysis...")
            progress_bar.progress(50)
            
            # Initialize variables
            semantic_max = semantic_avg = top_3_avg = tfidf_similarity = keyword_overlap_pct = skills_match = 0
            overlapping_keywords = set()
            
            if analysis_type in ["Composite Score (Recommended)", "Semantic Only"]:
                semantic_max, semantic_avg, top_3_avg = calculate_semantic_similarity(resume_clean, job_clean)
            
            progress_bar.progress(65)
            
            if analysis_type in ["Composite Score (Recommended)", "TF-IDF Only"]:
                tfidf_similarity = calculate_tfidf_similarity(resume_clean, job_clean)
            
            progress_bar.progress(75)
            
            if analysis_type in ["Composite Score (Recommended)", "Keyword Only"]:
                resume_keywords = extract_keywords(resume_clean, max_features=150)
                job_keywords = extract_keywords(job_clean, max_features=150)
                keyword_overlap_pct, overlapping_keywords = calculate_keyword_overlap(resume_keywords, job_keywords)
            
            progress_bar.progress(85)
            
            # Calculate skills match
            if analysis_type == "Composite Score (Recommended)":
                skills_match = calculate_skills_match(resume_text, job_desc)

            status_text.text("📊 Calculating final scores...")
            progress_bar.progress(95)

            # Calculate final score
            if analysis_type == "Composite Score (Recommended)":
                final_score = calculate_composite_score(
                    semantic_max, semantic_avg, top_3_avg, 
                    tfidf_similarity, keyword_overlap_pct, skills_match
                )
                score_type = "Composite"
            elif analysis_type == "Semantic Only":
                final_score = ((semantic_max * 0.5 + top_3_avg * 0.3 + semantic_avg * 0.2) * 100)
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
        
        st.info(f"📝 **Analysis Result:** {message}")

        # Detailed analysis section
        if show_details:
            st.markdown("---")
            st.markdown("## 📊 Detailed Analysis")
            
            # Create metrics in a beautiful layout
            if analysis_type == "Composite Score (Recommended)":
                metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
                
                with metric_col1:
                    st.markdown(
                        get_metric_card_html(f"{semantic_max*100:.1f}%", "Best Match"),
                        unsafe_allow_html=True
                    )
                
                with metric_col2:
                    st.markdown(
                        get_metric_card_html(f"{top_3_avg*100:.1f}%", "Top 3 Avg"),
                        unsafe_allow_html=True
                    )
                
                with metric_col3:
                    st.markdown(
                        get_metric_card_html(f"{tfidf_similarity*100:.1f}%", "TF-IDF"),
                        unsafe_allow_html=True
                    )
                
                with metric_col4:
                    st.markdown(
                        get_metric_card_html(f"{keyword_overlap_pct:.1f}%", "Keywords"),
                        unsafe_allow_html=True
                    )
                
                with metric_col5:
                    st.markdown(
                        get_metric_card_html(f"{skills_match*100:.1f}%", "Skills"),
                        unsafe_allow_html=True
                    )
            else:
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.markdown(
                        get_metric_card_html(f"{semantic_max*100:.1f}%", "Best Semantic"),
                        unsafe_allow_html=True
                    )
                
                with metric_col2:
                    st.markdown(
                        get_metric_card_html(f"{tfidf_similarity*100:.1f}%", "TF-IDF"),
                        unsafe_allow_html=True
                    )
                
                with metric_col3:
                    st.markdown(
                        get_metric_card_html(f"{keyword_overlap_pct:.1f}%", "Keywords"),
                        unsafe_allow_html=True
                    )

            # Show matching keywords if available
            if overlapping_keywords and len(overlapping_keywords) > 0:
                st.markdown("### 🎯 Matching Keywords")
                keywords_display = ", ".join(sorted(list(overlapping_keywords))[:30])
                st.success(f"**Found {len(overlapping_keywords)} matching keywords:** {keywords_display}")

            # Score visualization
            if show_visualization and analysis_type == "Composite Score (Recommended)":
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📈 Score Breakdown")
                scores_dict = {
                    "Best Match": semantic_max * 100,
                    "Top 3 Avg": top_3_avg * 100,
                    "TF-IDF": tfidf_similarity * 100,
                    "Keywords": keyword_overlap_pct,
                    "Skills": skills_match * 100
                }
                fig = create_score_visualization(scores_dict)
                st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(get_footer_html(), unsafe_allow_html=True)