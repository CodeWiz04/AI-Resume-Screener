

def get_custom_css(theme="Light"):
    """
    Returns enhanced custom CSS based on theme selection
    """
    
    # Theme color palettes
    if theme == "Light":
        colors = {
            'primary': '#6366f2',
            'secondary': '#8b5cf6',
            'accent': '#ec4899',
            'success': '#10b981',
            'warning': '#f59e0b',
            'danger': '#ef4444',
            'bg': '#ffffff',
            'bg_secondary': '#f8fafc',
            'text': '#1e293b',
            'text_secondary': '#64748b',
            'border': '#e2e8f0',
            'shadow': 'rgba(0, 0, 0, 0.1)',
            'card_bg': '#ffffff'
        }
    else:
        colors = {
            'primary': '#818cf8',
            'secondary': '#a78bfa',
            'accent': '#f472b6',
            'success': '#34d399',
            'warning': '#fbbf24',
            'danger': '#f87171',
            'bg': '#0f172a',
            'bg_secondary': '#1e293b',
            'text': '#f1f5f9',
            'text_secondary': '#94a3b8',
            'border': '#334155',
            'shadow': 'rgba(0, 0, 0, 0.3)',
            'card_bg': '#1e293b'
        }
    
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {{
        font-family: 'Inter', sans-serif;
    }}
    
    .main {{
        background: linear-gradient(135deg, {colors['bg']} 0%, {colors['bg_secondary']} 100%);
        color: {colors['text']};
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Enhanced Header with Animation */
    .header-container {{
        background: linear-gradient(135deg, {colors['primary']} 0%, {colors['secondary']} 50%, {colors['accent']} 100%);
        padding: 3rem 2rem;
        border-radius: 24px;
        margin-bottom: 2.5rem;
        text-align: center;
        box-shadow: 0 20px 60px {colors['shadow']};
        position: relative;
        overflow: hidden;
        animation: fadeInDown 0.8s ease-out;
    }}
    
    .header-container::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shimmer 3s infinite;
    }}
    
    @keyframes shimmer {{
        0% {{ transform: translateX(-100%) translateY(-100%) rotate(45deg); }}
        100% {{ transform: translateX(100%) translateY(100%) rotate(45deg); }}
    }}
    
    @keyframes fadeInDown {{
        from {{
            opacity: 0;
            transform: translateY(-30px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    .header-title {{
        font-size: 3.5rem;
        font-weight: 800;
        color: white;
        margin: 0;
        text-shadow: 0 4px 20px rgba(0,0,0,0.2);
        letter-spacing: -1px;
        position: relative;
        z-index: 1;
    }}
    
    .header-subtitle {{
        font-size: 1.2rem;
        color: rgba(255,255,255,0.95);
        margin-top: 1rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }}
    
    /* Score Display with Gradient Border */
    .score-container {{
        padding: 3rem;
        border-radius: 24px;
        margin: 2rem 0;
        text-align: center;
        position: relative;
        box-shadow: 0 20px 60px {colors['shadow']};
        animation: scaleIn 0.6s ease-out;
    }}
    
    @keyframes scaleIn {{
        from {{
            opacity: 0;
            transform: scale(0.9);
        }}
        to {{
            opacity: 1;
            transform: scale(1);
        }}
    }}
    
    .score-excellent {{ 
        background: linear-gradient(135deg, #10b981 0%, #34d399 50%, #6ee7b7 100%);
        color: white;
    }}
    
    .score-good {{ 
        background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 50%, #93c5fd 100%);
        color: white;
    }}
    
    .score-moderate {{ 
        background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 50%, #fcd34d 100%);
        color: #1e293b;
    }}
    
    .score-low {{ 
        background: linear-gradient(135deg, #ef4444 0%, #f87171 50%, #fca5a5 100%);
        color: white;
    }}
    
    .score-text {{
        font-size: 4rem;
        font-weight: 900;
        margin: 0;
        text-shadow: 0 4px 20px rgba(0,0,0,0.2);
        animation: pulse 2s infinite;
    }}
    
    @keyframes pulse {{
        0%, 100% {{ transform: scale(1); }}
        50% {{ transform: scale(1.05); }}
    }}
    
    .score-description {{
        font-size: 1.3rem;
        margin-top: 1rem;
        font-weight: 500;
        opacity: 0.95;
    }}
    
    /* Enhanced Metric Cards */
    .metric-card {{
        background: {colors['card_bg']};
        padding: 2rem 1.5rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px {colors['shadow']};
        border: 1px solid {colors['border']};
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
        animation: fadeIn 0.6s ease-out backwards;
    }}
    
    .metric-card:nth-child(1) {{ animation-delay: 0.1s; }}
    .metric-card:nth-child(2) {{ animation-delay: 0.2s; }}
    .metric-card:nth-child(3) {{ animation-delay: 0.3s; }}
    .metric-card:nth-child(4) {{ animation-delay: 0.4s; }}
    
    @keyframes fadeIn {{
        from {{
            opacity: 0;
            transform: translateY(20px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    .metric-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, {colors['primary']}, {colors['accent']});
        transform: scaleX(0);
        transition: transform 0.4s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-8px);
        box-shadow: 0 20px 50px {colors['shadow']};
    }}
    
    .metric-card:hover::before {{
        transform: scaleX(1);
    }}
    
    .metric-value {{
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, {colors['primary']}, {colors['accent']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.5rem 0;
    }}
    
    .metric-label {{
        font-size: 0.95rem;
        color: {colors['text_secondary']};
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    /* Enhanced Upload Area */
    .stFileUploader {{
        border: none !important;
        background: transparent !important;
        padding: 0 !important;
    }}
    
    .stFileUploader > div {{
        border: none !important;
        background: transparent !important;
    }}
    
    .stFileUploader > div > div {{
        background: linear-gradient(145deg, {colors['card_bg']}, {colors['bg_secondary']}) !important;
        border: 3px dashed {colors['border']} !important;
        border-radius: 24px !important;
        padding: 3rem 2rem !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        box-shadow: 0 10px 30px {colors['shadow']} !important;
        position: relative !important;
        overflow: hidden !important;
        min-height: 280px !important;
    }}
    
    .stFileUploader > div > div::before {{
        content: '📄';
        position: absolute;
        font-size: 4rem;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -60%);
        opacity: 0.1;
        z-index: 0;
    }}
    
    .stFileUploader > div > div:hover {{
        border-color: {colors['primary']} !important;
        background: linear-gradient(145deg, {colors['primary']}15, {colors['secondary']}15) !important;
        transform: translateY(-5px) !important;
        box-shadow: 0 15px 40px {colors['shadow']} !important;
        border-style: solid !important;
    }}
    
    .stFileUploader label {{
        display: none !important;
    }}
    
    .stFileUploader button {{
        background: linear-gradient(135deg, {colors['primary']}, {colors['secondary']}) !important;
        color: white !important;
        border: none !important;
        padding: 0.8rem 2rem !important;
        border-radius: 50px !important;
        font-weight: 600 !important;
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.3) !important;
        transition: all 0.3s ease !important;
        z-index: 1 !important;
        position: relative !important;
    }}
    
    .stFileUploader button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 25px rgba(99, 102, 241, 0.4) !important;
    }}
    
    .stFileUploader small {{
        color: {colors['text_secondary']} !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }}
    
    /* Enhanced Text Area */
    .stTextArea {{
        height: 100% !important;
    }}
    
    .stTextArea > div {{
        height: 100% !important;
    }}
    
    .stTextArea > div > div {{
        height: 100% !important;
    }}
    
    .stTextArea textarea {{
        border-radius: 24px !important;
        border: 3px solid {colors['border']} !important;
        padding: 1.5rem !important;
        font-size: 1rem !important;
        background: linear-gradient(145deg, {colors['card_bg']}, {colors['bg_secondary']}) !important;
        color: {colors['text']} !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        box-shadow: 0 10px 30px {colors['shadow']} !important;
        line-height: 1.6 !important;
        font-family: 'Inter', sans-serif !important;
        min-height: 280px !important;
        resize: none !important;
    }}
    
    .stTextArea textarea::placeholder {{
        color: {colors['text_secondary']} !important;
        opacity: 0.6 !important;
    }}
    
    .stTextArea textarea:focus {{
        border-color: {colors['primary']} !important;
        box-shadow: 0 15px 40px {colors['shadow']}, 0 0 0 4px {colors['primary']}20 !important;
        transform: translateY(-3px) !important;
        background: {colors['card_bg']} !important;
    }}
    
    .stTextArea textarea:hover {{
        border-color: {colors['primary']}80 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 35px {colors['shadow']} !important;
    }}
    
    /* Enhanced Button */
    .stButton > button {{
        background: linear-gradient(135deg, {colors['primary']} 0%, {colors['accent']} 100%) !important;
        color: white !important;
        border: none !important;
        padding: 1.2rem 3rem !important;
        border-radius: 50px !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.4) !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        position: relative;
        overflow: hidden;
    }}
    
    .stButton > button::before {{
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255,255,255,0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }}
    
    .stButton > button:hover::before {{
        width: 300px;
        height: 300px;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-5px) scale(1.05) !important;
        box-shadow: 0 15px 40px rgba(99, 102, 241, 0.6) !important;
    }}
    
    .stButton > button:active {{
        transform: translateY(-2px) scale(1.02) !important;
    }}
    
    /* Success Upload Message */
    .upload-success {{
        background: linear-gradient(135deg, {colors['success']} 0%, #34d399 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 16px;
        margin-top: 1rem;
        text-align: center;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3);
        animation: slideInUp 0.5s ease-out;
    }}
    
    @keyframes slideInUp {{
        from {{
            opacity: 0;
            transform: translateY(20px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    /* Keyword Tags */
    .keyword-tag {{
        display: inline-block;
        background: linear-gradient(135deg, {colors['primary']}, {colors['secondary']});
        color: white;
        padding: 0.6rem 1.2rem;
        margin: 0.4rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 600;
        box-shadow: 0 4px 15px {colors['shadow']};
        transition: all 0.3s ease;
        animation: fadeIn 0.4s ease-out backwards;
    }}
    
    .keyword-tag:hover {{
        transform: translateY(-3px);
        box-shadow: 0 6px 20px {colors['shadow']};
    }}
    
    .missing-keyword-tag {{
        display: inline-block;
        background: linear-gradient(135deg, {colors['warning']}, {colors['danger']});
        color: white;
        padding: 0.6rem 1.2rem;
        margin: 0.4rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 600;
        box-shadow: 0 4px 15px {colors['shadow']};
        transition: all 0.3s ease;
        animation: fadeIn 0.4s ease-out backwards;
    }}
    
    .missing-keyword-tag:hover {{
        transform: translateY(-3px);
        box-shadow: 0 6px 20px {colors['shadow']};
    }}
    
    /* Progress Bar */
    .stProgress > div > div > div {{
        background: linear-gradient(90deg, {colors['primary']}, {colors['accent']}) !important;
        border-radius: 10px !important;
    }}
    
    /* Info/Warning/Error Boxes */
    .stAlert {{
        border-radius: 16px !important;
        border: none !important;
        box-shadow: 0 8px 25px {colors['shadow']} !important;
        animation: slideInLeft 0.5s ease-out;
    }}
    
    @keyframes slideInLeft {{
        from {{
            opacity: 0;
            transform: translateX(-20px);
        }}
        to {{
            opacity: 1;
            transform: translateX(0);
        }}
    }}
    
    /* Sidebar Styling */
    .css-1d391kg {{
        background: {colors['card_bg']};
        border-right: 1px solid {colors['border']};
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
        background: {colors['card_bg']} !important;
        border-radius: 12px !important;
        border: 1px solid {colors['border']} !important;
        font-weight: 600 !important;
        color: {colors['text']} !important;
    }}
    
    /* Section Headers */
    h1, h2, h3 {{
        color: {colors['text']} !important;
        font-weight: 700 !important;
    }}
    
    h2 {{
        border-bottom: 3px solid {colors['primary']};
        padding-bottom: 0.5rem;
        display: inline-block;
    }}
    
    /* Selectbox */
    .stSelectbox {{
        border-radius: 12px;
    }}
    
    /* Checkbox */
    .stCheckbox {{
        font-weight: 500;
    }}
    
    /* Loading Spinner */
    .stSpinner > div {{
        border-top-color: {colors['primary']} !important;
    }}
    
    /* Footer */
    .footer {{
        text-align: center;
        padding: 3rem 2rem;
        color: {colors['text_secondary']};
        font-size: 0.95rem;
        border-top: 1px solid {colors['border']};
        margin-top: 4rem;
    }}
    
    /* Responsive Design */
    @media (max-width: 768px) {{
        .header-title {{
            font-size: 2rem;
        }}
        
        .header-subtitle {{
            font-size: 1rem;
        }}
        
        .score-text {{
            font-size: 3rem;
        }}
        
        .metric-value {{
            font-size: 2rem;
        }}
    }}
    </style>
    """


def get_header_html():
    """Returns the enhanced header HTML"""
    return """
    <div class="header-container">
        <h1 class="header-title">🎯 AI Resume Screener</h1>
        <p class="header-subtitle">Advanced AI-powered resume analysis with comprehensive matching algorithms</p>
    </div>
    """


def get_upload_success_html(filename, filesize):
    """Returns success message HTML for file upload"""
    return f"""
    <div class="upload-success">
        ✅ <strong>{filename}</strong> uploaded successfully!<br>
        <small>File size: {filesize / 1024:.1f} KB</small>
    </div>
    """


def get_score_html(score, score_type):
    """Returns the score display HTML with appropriate styling"""
    if score >= 70:
        score_class = "score-excellent"
        emoji = "🎉"
        message = "Excellent match! Your resume aligns perfectly with the job requirements."
    elif score >= 50:
        score_class = "score-good"
        emoji = "✅"
        message = "Good match with room for improvement. Consider adding more relevant keywords."
    elif score >= 30:
        score_class = "score-moderate"
        emoji = "⚠️"
        message = "Moderate match. Your resume could benefit from significant tailoring."
    else:
        score_class = "score-low"
        emoji = "🔄"
        message = "Low match. Consider restructuring your resume to better align with this job."
    
    return f"""
    <div class="score-container {score_class}">
        <p class="score-text">{emoji} {score}%</p>
        <p class="score-description">{score_type} Match Score</p>
    </div>
    """, message


def get_metric_card_html(value, label):
    """Returns a metric card HTML"""
    return f"""
    <div class="metric-card">
        <p class="metric-value">{value}</p>
        <p class="metric-label">{label}</p>
    </div>
    """


def get_section_header_html(icon, title):
    """Returns styled section header HTML"""
    return f"""
    <div style="
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1.5rem;
        padding: 1rem 1.5rem;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1));
        border-radius: 16px;
        border-left: 4px solid #6366f1;
    ">
        <span style="font-size: 2rem;">{icon}</span>
        <h3 style="
            margin: 0;
            font-size: 1.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        ">{title}</h3>
    </div>
    """


def get_footer_html():
    """Returns the footer HTML"""
    return """
    <div class="footer">
        <p>Made with ❤️ using Streamlit • Advanced AI Resume Analysis • Version 2.0</p>
        <p style="margin-top: 0.5rem; font-size: 0.85rem;">Powered by state-of-the-art NLP and Machine Learning</p>
    </div>
    """