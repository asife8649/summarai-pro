# ⚡ SummarAI Pro

> Beautiful AI text summarizer — TF-IDF algorithm, 7 languages, PDF/DOCX upload, 10,000-word limit.  
> **Zero API keys required. 100% local.**

---

## ✨ Features

| Feature | Detail |
|---------|--------|
| 🧠 Algorithm | TF-IDF Extractive Summarization (scikit-learn) |
| 📏 Word Limit | 10,000 words per input |
| 🌐 Languages | English, Tamil, Hindi, French, Spanish, German, Arabic, Bengali, Japanese |
| 📂 File Upload | PDF, DOCX, TXT |
| 📊 Analysis Tab | Keywords, readability score, word frequency chart |
| ⬇️ Download | Export summary as .txt |

---

## 🚀 Setup in VS Code

### 1. Open the project folder
```
File → Open Folder → select summarizer_pro
```

### 2. Open Terminal
```
Ctrl + `
```

### 3. Create virtual environment
```bash
python -m venv venv
```

Activate it:
```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 4. Install all dependencies
```bash
pip install -r requirements.txt
```

### 5. Download NLTK data (one time)
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

### 6. Run the app
```bash
streamlit run app.py
```

Open browser at **http://localhost:8501**

---

## 📁 Project Structure

```
summarizer_pro/
├── app.py            ← Streamlit UI (beautiful dark theme)
├── core.py           ← TF-IDF summarization + keywords + readability
├── translator.py     ← 7-language translation (deep-translator)
├── file_reader.py    ← PDF / DOCX / TXT file extraction
├── requirements.txt  ← All dependencies
└── README.md
```

---

## 🧠 How TF-IDF Works

1. **Tokenize** — Split text into sentences
2. **Vectorize** — Build TF-IDF matrix (sentences × unique terms)
3. **Score** — Each sentence score = sum of its TF-IDF values
4. **Rank** — Pick top-N sentences by score
5. **Order** — Return them in original reading order

---

## 🌐 Supported Languages

| Language | Code |
|----------|------|
| English  | en   |
| Tamil    | ta   |
| Hindi    | hi   |
| French   | fr   |
| Spanish  | es   |
| German   | de   |
| Arabic   | ar   |
| Bengali  | bn   |
| Japanese | ja   |

Translation uses `deep-translator` (Google Translate backend) — free, no API key needed.

---

## ⚠️ Common Issues

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside venv |
| NLTK errors | Run the NLTK download command in step 5 |
| PDF text empty | Install PyMuPDF: `pip install PyMuPDF` |
| DOCX not working | Install: `pip install python-docx` |
| Translation fails | Check internet connection (uses Google Translate) |
