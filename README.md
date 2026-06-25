# 🎯 AI Resume Screener

An intelligent, multi-strategy resume screening tool built with **Streamlit** and **Python** that analyzes how well a resume matches a job description using semantic similarity, keyword overlap, TF-IDF analysis, and technical skill matching.

---

## ✨ Features

- **📄 Multi-format Resume Support** — Upload resumes in PDF, DOCX, or TXT format
- **🤖 Semantic Similarity Analysis** — Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to understand the *meaning* behind resume content, not just keywords
- **🔍 Exhaustive Keyword Matching** — Extracts single words, bigrams, trigrams, and up to 5-word phrases for maximum coverage
- **📊 TF-IDF Content Analysis** — Vectorizes both documents with n-grams (1–5) to measure content relevance
- **⚙️ Technical Skills Extractor** — Detects 150+ technologies across languages, frameworks, databases, cloud tools, AI/ML, and more
- **🧩 Composite Scoring Engine** — Combines all signals with an aggressive boost curve for a final match percentage
- **📈 Radar Chart Visualization** — Interactive Plotly radar chart breaking down each score dimension
- **🎨 Light / Dark Theme** — Toggle between themes from the sidebar
- **💡 Match Insights** — Actionable feedback on semantic alignment, keyword coverage, and technical gaps

---

## 🖥️ Demo

> Upload your resume → Paste a job description → Click **Analyze Resume Match** → Get a comprehensive score with breakdown

---

## 🗂️ Project Structure

```
AI-Resume-Screener/
├── main.py          # Core Streamlit app — text extraction, scoring, UI
├── styles.py        # Custom HTML/CSS components for UI theming
├── .gitignore
├── LICENSE          # MIT License
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/CodeWiz04/AI-Resume-Screener.git
cd AI-Resume-Screener

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** If there's no `requirements.txt` yet, install the packages manually:
> ```bash
> pip install streamlit pymupdf python-docx sentence-transformers scikit-learn numpy plotly nltk
> ```

### Running the App

```bash
streamlit run main.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `pymupdf` (`fitz`) | PDF text extraction |
| `python-docx` | DOCX text extraction |
| `sentence-transformers` | Semantic similarity via `all-MiniLM-L6-v2` |
| `scikit-learn` | TF-IDF vectorization & cosine similarity |
| `numpy` | Numerical computations |
| `plotly` | Interactive radar chart |
| `nltk` | Tokenization, stopwords, lemmatization |

---

## 🧠 How It Works

The scoring pipeline combines four independent signals:

### 1. Semantic Similarity
The resume and job description are split into overlapping chunks (small, medium, large, sentence-level, and full text). Each chunk is encoded using `all-MiniLM-L6-v2` and compared via cosine similarity. The best-match, top-3, top-5, and top-10 chunk similarities are extracted.

### 2. Keyword Overlap
All possible 1–5 word n-grams are extracted from both documents. Matching happens in 5 levels — exact match → substring → word overlap → character similarity → stem/root matching — giving maximum possible coverage.

### 3. TF-IDF Analysis
Both texts are vectorized using TF-IDF with n-grams from 1 to 5 words and up to 5,000 features. The cosine similarity of the resulting vectors measures vocabulary and phrasing alignment.

### 4. Technical Skills Matching
A comprehensive regex-based extractor identifies technologies across 10 categories (languages, frameworks, databases, cloud, DevOps tools, AI/ML, mobile, web, and more). Partial matches receive 70% credit.

### Composite Score Formula
```
score = (semantic_max × 0.20) + (semantic_top3 × 0.18) + (semantic_top5 × 0.12)
      + (semantic_top10 × 0.08) + (tfidf × 0.22) + (keywords × 0.15)
      + (tech_match × 0.05)
```
A length bonus and an aggressive boost curve are then applied to produce the final percentage.

---

## 🎛️ Analysis Modes

| Mode | Description |
|---|---|
| **Composite Score** *(Recommended)* | Combines all four strategies for the most accurate result |
| **Semantic Only** | Uses only sentence-transformer embeddings |
| **Keyword Only** | Uses only n-gram keyword overlap |
| **TF-IDF Only** | Uses only TF-IDF vectorization |

---

## 📊 Score Interpretation

| Score | Rating |
|---|---|
| 🟢 70–100% | Excellent match |
| 🟡 50–69% | Good match |
| 🟠 30–49% | Moderate match |
| 🔴 0–29% | Low match |

---

## 💡 Tips for Best Results

- Paste the **complete** job description including responsibilities, qualifications, and required skills
- Use **exact terminology** from the job posting in your resume
- Include **quantified achievements** (e.g., "Improved performance by 40%")
- List all **technical skills** explicitly — the extractor looks for specific terms
- Longer, more detailed resumes (300+ words) receive a small length bonus

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**CodeWiz04** — [GitHub Profile](https://github.com/CodeWiz04)

---

> ⭐ If you found this project useful, consider giving it a star on GitHub!
