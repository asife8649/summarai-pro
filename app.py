import streamlit as st
import time
import re
from pathlib import Path
from core import tfidf_summarize, extract_keywords, readability_score
from translator import translate_summary
from file_reader import read_uploaded_file

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SummarAI Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

WORD_LIMIT = 10000

LANGUAGES = {
    "English":    "en",
    "Tamil":      "ta",
    "Hindi":      "hi",
    "French":     "fr",
    "Spanish":    "es",
    "German":     "de",
    "Arabic":     "ar",
    "Bengali":    "bn",
    "Japanese":   "ja",
}

# ── Inject CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] { font-family: 'Outfit', sans-serif !important; }

.stApp {
    background: #080b14;
    color: #c8d0e7;
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1020 !important;
    border-right: 1px solid #1a2040 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1020; }
::-webkit-scrollbar-thumb { background: #2a3560; border-radius: 3px; }

/* ── Navbar ── */
.navbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1rem 2rem;
    border-bottom: 1px solid #1a2040;
    background: rgba(8, 11, 20, 0.95);
    position: sticky; top: 0; z-index: 100;
    backdrop-filter: blur(12px);
}
.navbar-brand {
    display: flex; align-items: center; gap: 0.75rem;
}
.brand-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #6c63ff, #00d4ff);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
}
.brand-name {
    font-size: 1.4rem; font-weight: 800;
    background: linear-gradient(135deg, #6c63ff, #00d4ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
}
.brand-badge {
    background: linear-gradient(135deg, #6c63ff22, #00d4ff22);
    border: 1px solid #6c63ff55;
    color: #00d4ff;
    font-size: 0.65rem; font-weight: 600;
    padding: 2px 8px; border-radius: 20px;
    letter-spacing: 1px; text-transform: uppercase;
}
.nav-stats {
    display: flex; gap: 1.5rem;
}
.nav-stat {
    text-align: center;
}
.nav-stat-val {
    font-size: 1rem; font-weight: 700;
    background: linear-gradient(135deg, #6c63ff, #00d4ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.nav-stat-label {
    font-size: 0.65rem; color: #4a5580; text-transform: uppercase; letter-spacing: 1px;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 3rem 2rem 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: -60px; left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 300px;
    background: radial-gradient(ellipse, #6c63ff18 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.8rem; font-weight: 800;
    color: #ffffff; line-height: 1.1;
    letter-spacing: -1.5px;
    margin-bottom: 0.8rem;
}
.hero-title span {
    background: linear-gradient(135deg, #6c63ff, #00d4ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 1rem; color: #4a5580; font-weight: 300;
    letter-spacing: 0.5px;
}
.algo-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: #0d1020; border: 1px solid #1a2040;
    border-radius: 20px; padding: 6px 14px;
    font-size: 0.75rem; color: #6c63ff; font-weight: 500;
    margin-top: 1rem; letter-spacing: 0.5px;
}
.algo-dot { width: 6px; height: 6px; border-radius: 50%; background: #00d4ff; }

/* ── Main layout ── */
.main-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    padding: 1.5rem 2rem;
    max-width: 1400px; margin: 0 auto;
}

/* ── Panel cards ── */
.panel {
    background: #0d1020;
    border: 1px solid #1a2040;
    border-radius: 20px;
    overflow: hidden;
}
.panel-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1.2rem 1.5rem;
    border-bottom: 1px solid #1a2040;
    background: #0a0e1a;
}
.panel-title {
    display: flex; align-items: center; gap: 0.6rem;
    font-size: 1rem; font-weight: 600; color: #fff;
}
.panel-icon {
    width: 30px; height: 30px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem;
}
.panel-icon.blue { background: #6c63ff22; }
.panel-icon.cyan { background: #00d4ff22; }
.panel-body { padding: 1.5rem; }

/* ── Word counter ── */
.word-counter {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 0.8rem;
}
.wc-text { font-size: 0.8rem; color: #4a5580; }
.wc-badge {
    font-size: 0.75rem; font-weight: 600;
    padding: 3px 10px; border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
}
.wc-ok   { background: #00d4ff15; color: #00d4ff; border: 1px solid #00d4ff33; }
.wc-warn { background: #ffaa0015; color: #ffaa00; border: 1px solid #ffaa0033; }
.wc-over { background: #ff444415; color: #ff4444; border: 1px solid #ff444433; }

/* ── Progress bar ── */
.progress-wrap {
    background: #131826; border-radius: 4px;
    height: 4px; margin-bottom: 1rem; overflow: hidden;
}
.progress-bar {
    height: 100%; border-radius: 4px;
    transition: width 0.3s ease, background 0.3s ease;
}

/* ── Settings row ── */
.settings-row {
    display: flex; gap: 1rem; margin-bottom: 1.2rem;
    padding: 1rem 1.5rem;
    background: #0a0e1a;
    border-top: 1px solid #1a2040;
    border-bottom: 1px solid #1a2040;
}
.setting-item { flex: 1; }
.setting-label {
    font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 1px; color: #4a5580;
    margin-bottom: 4px; font-weight: 500;
}

/* ── Summary result ── */
.summary-result {
    background: #080b14;
    border: 1px solid #1a2040;
    border-left: 3px solid #6c63ff;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    font-size: 0.95rem;
    line-height: 1.9;
    color: #c8d0e7;
    font-weight: 300;
    min-height: 120px;
}
.summary-placeholder {
    color: #2a3560; font-style: italic;
    display: flex; align-items: center; justify-content: center;
    height: 120px; font-size: 0.9rem;
}

/* ── Stats grid ── */
.stats-grid {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 0.8rem; margin-bottom: 1.2rem;
}
.stat-card {
    background: #080b14;
    border: 1px solid #1a2040;
    border-radius: 12px;
    padding: 0.9rem;
    text-align: center;
}
.stat-val {
    font-size: 1.6rem; font-weight: 700;
    background: linear-gradient(135deg, #6c63ff, #00d4ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    font-family: 'JetBrains Mono', monospace;
}
.stat-label {
    font-size: 0.68rem; text-transform: uppercase;
    letter-spacing: 1px; color: #4a5580; margin-top: 2px;
}

/* ── Keywords ── */
.kw-wrap { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 1.2rem; }
.kw-chip {
    background: #131826; border: 1px solid #1a2040;
    border-radius: 6px; padding: 3px 10px;
    font-size: 0.75rem; color: #6c63ff;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Language chips ── */
.lang-note {
    font-size: 0.75rem; color: #4a5580;
    background: #0a0e1a; border: 1px solid #1a2040;
    border-radius: 8px; padding: 0.5rem 0.8rem;
    display: flex; align-items: center; gap: 6px;
    margin-bottom: 1rem;
}

/* ── Error box ── */
.error-box {
    background: #ff444415; border: 1px solid #ff444433;
    border-radius: 10px; padding: 1rem; color: #ff6b6b;
    font-size: 0.85rem; margin-bottom: 1rem;
}

/* ── Warning box ── */
.warn-box {
    background: #ffaa0015; border: 1px solid #ffaa0033;
    border-radius: 10px; padding: 1rem; color: #ffcc44;
    font-size: 0.85rem; margin-bottom: 1rem;
}

/* ── Streamlit widget overrides ── */
.stTextArea textarea {
    background: #080b14 !important;
    border: 1px solid #1a2040 !important;
    border-radius: 12px !important;
    color: #c8d0e7 !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.9rem !important;
    line-height: 1.7 !important;
    resize: vertical !important;
}
.stTextArea textarea:focus {
    border-color: #6c63ff !important;
    box-shadow: 0 0 0 3px #6c63ff15 !important;
}
.stSelectbox > div > div {
    background: #080b14 !important;
    border: 1px solid #1a2040 !important;
    border-radius: 10px !important;
    color: #c8d0e7 !important;
}
.stSlider > div > div > div {
    background: #6c63ff !important;
}
.stButton > button {
    background: linear-gradient(135deg, #6c63ff, #00d4ff) !important;
    color: #fff !important; border: none !important;
    border-radius: 12px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.9rem !important; font-weight: 600 !important;
    padding: 0.65rem 1.5rem !important;
    letter-spacing: 0.3px !important;
    transition: opacity 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
.stDownloadButton > button {
    background: #131826 !important;
    color: #6c63ff !important;
    border: 1px solid #6c63ff44 !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    width: 100% !important;
}
[data-testid="stFileUploader"] {
    background: #080b14 !important;
    border: 1px dashed #1a2040 !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"] * { color: #4a5580 !important; }

/* Tabs */
[data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1a2040 !important;
    gap: 0 !important;
}
[data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.82rem !important; font-weight: 500 !important;
    letter-spacing: 0.5px !important; text-transform: uppercase !important;
    color: #4a5580 !important;
    background: transparent !important;
    border: none !important;
    padding: 0.7rem 1.2rem !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: #6c63ff !important;
    border-bottom: 2px solid #6c63ff !important;
}
[data-testid="stTabContent"] { padding: 1.5rem 0 0 !important; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def count_words(text): return len(text.split()) if text.strip() else 0
def count_chars(text): return len(text)

def word_badge_class(wc):
    pct = wc / WORD_LIMIT
    if pct < 0.75:   return "wc-ok"
    if pct < 1.0:    return "wc-warn"
    return "wc-over"

def progress_color(wc):
    pct = wc / WORD_LIMIT
    if pct < 0.75:   return "#00d4ff"
    if pct < 1.0:    return "#ffaa00"
    return "#ff4444"

# ── Navbar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
  <div class="navbar-brand">
    <div class="brand-icon">⚡</div>
    <div class="brand-name">SummarAI</div>
    <div class="brand-badge">Pro</div>
  </div>
  <div class="nav-stats">
    <div class="nav-stat">
      <div class="nav-stat-val">TF-IDF</div>
      <div class="nav-stat-label">Algorithm</div>
    </div>
    <div class="nav-stat">
      <div class="nav-stat-val">10K</div>
      <div class="nav-stat-label">Word Limit</div>
    </div>
    <div class="nav-stat">
      <div class="nav-stat-val">9</div>
      <div class="nav-stat-label">Languages</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">Summarize Any Text<br><span>Instantly & Intelligently</span></div>
  <div class="hero-sub">Powered by TF-IDF · Supports 9 languages · PDF & DOCX upload</div>
  <div class="algo-pill">
    <div class="algo-dot"></div>
    TF-IDF Extractive Algorithm
  </div>
</div>
""", unsafe_allow_html=True)

# ── Main Two-Column Layout ────────────────────────────────────────────────────
col_in, col_out = st.columns([1, 1], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN — INPUT
# ══════════════════════════════════════════════════════════════════════════════
with col_in:
    st.markdown("""
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">
          <div class="panel-icon blue">📄</div>
          Input Text
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs: Type / Upload ────────────────────────────────────────────────
    tab_type, tab_upload = st.tabs(["✏️  Type / Paste", "📂  Upload File"])

    input_text = ""

    with tab_type:
        raw_input = st.text_area(
            "Enter your text",
            height=280,
            placeholder="Paste or type up to 10,000 words here…",
            label_visibility="collapsed",
            key="raw_text"
        )
        input_text = raw_input

    with tab_upload:
        uploaded = st.file_uploader(
            "Upload PDF or DOCX",
            type=["pdf", "docx", "txt"],
            label_visibility="collapsed",
            key="file_upload"
        )
        if uploaded:
            try:
                extracted = read_uploaded_file(uploaded)
                if extracted:
                    input_text = extracted
                    st.success(f"✅ File loaded — {count_words(extracted):,} words extracted")
                    with st.expander("Preview extracted text"):
                        st.write(extracted[:800] + ("…" if len(extracted) > 800 else ""))
            except Exception as e:
                st.markdown(f'<div class="error-box">❌ Could not read file: {e}</div>', unsafe_allow_html=True)

    # ── Word counter & progress bar ────────────────────────────────────────
    wc = count_words(input_text)
    pct_used = min(wc / WORD_LIMIT, 1.0)
    badge_cls = word_badge_class(wc)
    bar_color = progress_color(wc)

    st.markdown(f"""
    <div class="word-counter">
      <span class="wc-text">Word usage</span>
      <span class="wc-badge {badge_cls}">{wc:,} / {WORD_LIMIT:,} words</span>
    </div>
    <div class="progress-wrap">
      <div class="progress-bar" style="width:{pct_used*100:.1f}%; background:{bar_color};"></div>
    </div>
    """, unsafe_allow_html=True)

    if wc > WORD_LIMIT:
        st.markdown(f'<div class="warn-box">⚠️ Text exceeds {WORD_LIMIT:,}-word limit. Only the first {WORD_LIMIT:,} words will be processed.</div>', unsafe_allow_html=True)

    # ── Settings ──────────────────────────────────────────────────────────
    st.markdown("---")
    cfg1, cfg2 = st.columns(2)
    with cfg1:
        st.markdown('<div class="setting-label">Output Language</div>', unsafe_allow_html=True)
        out_lang = st.selectbox(
            "Language", list(LANGUAGES.keys()),
            label_visibility="collapsed", key="lang"
        )
    with cfg2:
        st.markdown('<div class="setting-label">Summary Length</div>', unsafe_allow_html=True)
        num_sents = st.slider(
            "Sentences", min_value=1, max_value=12, value=5,
            label_visibility="collapsed", key="sents"
        )

    summarize_btn = st.button("⚡ Generate Summary", key="go")

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — OUTPUT
# ══════════════════════════════════════════════════════════════════════════════
with col_out:
    st.markdown("""
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">
          <div class="panel-icon cyan">🧠</div>
          Summary Output
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    out_tab1, out_tab2 = st.tabs(["📋  Summary", "📊  Analysis"])

    with out_tab1:
        if summarize_btn:
            if not input_text.strip():
                st.markdown('<div class="error-box">❌ Please enter or upload some text first.</div>', unsafe_allow_html=True)
            elif wc < 20:
                st.markdown('<div class="warn-box">⚠️ Text too short. Please enter at least 20 words.</div>', unsafe_allow_html=True)
            else:
                # Enforce word limit
                words = input_text.split()
                if len(words) > WORD_LIMIT:
                    input_text = " ".join(words[:WORD_LIMIT])

                with st.spinner("Running TF-IDF algorithm…"):
                    time.sleep(0.4)
                    summary_en = tfidf_summarize(input_text, n=num_sents)

                # Translate if needed
                lang_code = LANGUAGES[out_lang]
                if lang_code != "en" and summary_en:
                    with st.spinner(f"Translating to {out_lang}…"):
                        final_summary = translate_summary(summary_en, lang_code)
                else:
                    final_summary = summary_en

                # Store
                st.session_state["summary"]    = final_summary
                st.session_state["input_snap"] = input_text
                st.session_state["lang_used"]  = out_lang

        # Show summary
        if "summary" in st.session_state and st.session_state["summary"]:
            s = st.session_state["summary"]
            inp = st.session_state.get("input_snap", "")
            sw = count_words(s)
            iw = count_words(inp)
            compression = round((1 - sw / max(iw, 1)) * 100, 1)

            # Stats
            st.markdown(f"""
            <div class="stats-grid">
              <div class="stat-card">
                <div class="stat-val">{iw:,}</div>
                <div class="stat-label">Original Words</div>
              </div>
              <div class="stat-card">
                <div class="stat-val">{sw}</div>
                <div class="stat-label">Summary Words</div>
              </div>
              <div class="stat-card">
                <div class="stat-val">{compression}%</div>
                <div class="stat-label">Compressed</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            lang_used = st.session_state.get("lang_used", "English")
            if lang_used != "English":
                st.markdown(f'<div class="lang-note">🌐 Translated to <strong>{lang_used}</strong></div>', unsafe_allow_html=True)

            st.markdown(f'<div class="summary-result">{s}</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                "⬇ Download Summary (.txt)",
                data=s,
                file_name="summary.txt",
                mime="text/plain",
            )
        else:
            st.markdown("""
            <div class="summary-result">
              <div class="summary-placeholder">✨ Your summary will appear here…</div>
            </div>
            """, unsafe_allow_html=True)

    with out_tab2:
        if "input_snap" in st.session_state:
            inp = st.session_state["input_snap"]

            # Keywords
            from core import extract_keywords
            kws = extract_keywords(inp, top_n=12)
            kw_html = " ".join(f'<span class="kw-chip">{k}</span>' for k in kws)
            st.markdown("**🔑 Top Keywords**")
            st.markdown(f'<div class="kw-wrap">{kw_html}</div>', unsafe_allow_html=True)

            # Readability
            score = readability_score(inp)
            if score >= 70:   read_label, read_color = "Easy", "#00d4ff"
            elif score >= 50: read_label, read_color = "Moderate", "#ffaa00"
            else:             read_label, read_color = "Complex", "#ff4444"

            st.markdown("**📖 Readability (Flesch)**")
            st.markdown(f"""
            <div class="stat-card" style="margin-bottom:1rem; text-align:left; padding:1rem;">
              <span style="font-size:1.8rem;font-weight:700;color:{read_color};
                font-family:'JetBrains Mono',monospace;">{score}</span>
              <span style="font-size:0.8rem;color:{read_color};margin-left:8px;">
                {read_label} to read</span>
            </div>
            """, unsafe_allow_html=True)

            # Word freq chart
            import re, pandas as pd
            from collections import Counter
            stopw = {"the","a","an","and","or","but","in","on","at","to","for","of",
                     "with","is","was","are","were","be","been","being","have","has",
                     "had","do","does","did","will","would","could","should","this",
                     "that","these","those","it","its","from","by","as","not","also"}
            wlist = [w.lower() for w in re.findall(r'\b[a-zA-Z]{4,}\b', inp) if w.lower() not in stopw]
            freq = Counter(wlist).most_common(8)
            if freq:
                df = pd.DataFrame(freq, columns=["Word","Count"]).set_index("Word")
                st.markdown("**📈 Word Frequency**")
                st.bar_chart(df, height=180)
        else:
            st.info("Run a summarization first to see analysis.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:2rem; color:#2a3560; font-size:0.75rem;
     border-top:1px solid #1a2040; margin-top:2rem;">
  SummarAI Pro · TF-IDF Extractive Summarization · 9 Languages · 10,000 Word Limit
</div>
""", unsafe_allow_html=True)
