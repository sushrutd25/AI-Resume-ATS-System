"""
Core functions of the AI Resume ATS, written so a web app can reuse them.

Pipeline: resume bytes -> pages (PyMuPDF) -> OpenCV preprocessing + layout checks
-> EasyOCR text -> keyword (TF-IDF) + semantic (Sentence-Transformers) scores.
"""
import re
import time

import cv2
import numpy as np
import pymupdf
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

MAX_PAGES = 3  # resumes are short; keeps OCR fast on CPU

SECTIONS = {
    "summary": r"\b(summary|objective|profile)\b",
    "experience": r"\b(experience|employment|work history|internship)\b",
    "education": r"\b(education|academic|qualification)\b",
    "skills": r"\b(skills|technologies|technical skills)\b",
    "projects": r"\b(projects?)\b",
}

EXTRA_STOP = {
    "job", "key", "related", "functional", "responsibilities",
    "requirements", "experience", "work", "working", "team", "teams",
    "ability", "strong", "skills", "years", "role", "candidate",
    "solutions", "preferred", "required", "including", "etc", "good",
    "excellent", "knowledge", "understanding", "looking", "join",
}
STOP = list(ENGLISH_STOP_WORDS.union(EXTRA_STOP))


# ------------------------------------------------------------ models
def load_reader():
    import easyocr  # imported here so this file loads fast
    return easyocr.Reader(["en"], gpu=False)


def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


# ------------------------------------------------------------ 1. load
def load_pages(data: bytes, filename: str):
    """Return a list of BGR images from PDF or image bytes."""
    if filename.lower().endswith(".pdf"):
        pages = []
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            for page in list(doc)[:MAX_PAGES]:
                pix = page.get_pixmap(dpi=200, alpha=False)
                img = np.frombuffer(pix.samples, dtype=np.uint8)
                img = img.reshape(pix.height, pix.width, 3)
                pages.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return pages
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return [img] if img is not None else []


# ------------------------------------------------------------ 2. preprocess
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, h=10)
    inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(inv > 0))
    if len(coords) > 100:
        angle = cv2.minAreaRect(coords)[-1]
        if angle > 45:
            angle -= 90
        if 0.5 < abs(angle) < 15:
            h, w = gray.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
    return gray


# ------------------------------------------------------------ 3. layout
def layout_report(gray):
    h, w = gray.shape
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 15, 10)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 15, 1), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 15, 1)))
    hl = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk)
    vl = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk)
    n_h = len(cv2.findContours(hl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
    n_v = len(cv2.findContours(vl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
    has_table = n_h >= 2 and n_v >= 2

    profile = (bw > 0).sum(axis=0)
    lo, hi = int(w * 0.30), int(w * 0.70)
    empty = profile[lo:hi] <= 0.005 * h
    longest = run = 0
    for e in empty:
        run = run + 1 if e else 0
        longest = max(longest, run)
    has_columns = longest >= w * 0.02
    return {"table": has_table, "columns": has_columns}


# ------------------------------------------------------------ 4. OCR
def run_ocr(reader, gray):
    results = reader.readtext(gray)
    text = " ".join(r[1] for r in results)
    conf = float(np.mean([r[2] for r in results])) if results else 0.0
    return text, conf, results


def annotate(img, results):
    """Green box = confident text, red box = low-confidence text."""
    out = img.copy()
    for bbox, _, conf in results:
        pts = np.array(bbox, dtype=np.int32)
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        color = (0, 180, 0) if conf > 0.6 else (0, 0, 255)
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    return out


# ------------------------------------------------------------ 5. text checks
def check_sections(text):
    t = text.lower()
    return {s: bool(re.search(p, t)) for s, p in SECTIONS.items()}


def check_contact(text):
    t = re.sub(r"\s*@\s*", "@", text)
    t = re.sub(r"\s*\.\s*(com|in|org|net|co|io)\b", r".\1", t, flags=re.I)
    email = bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", t))
    phone = bool(re.search(r"(\+91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}", text)) or \
        bool(re.search(r"\+?\d[\d\s\-()]{9,}\d", text))
    return {"email": email, "phone": phone}


# ------------------------------------------------------------ 6. scoring
def extract_keywords(jd_text, top_n=20):
    tfidf = TfidfVectorizer(stop_words=STOP, ngram_range=(1, 2),
                            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z+#]+\b",
                            max_features=top_n * 3)
    weights = tfidf.fit_transform([jd_text]).toarray()[0]
    terms = tfidf.get_feature_names_out()
    ranked = [t for _, t in sorted(zip(weights, terms), reverse=True)]
    bigrams = [t for t in ranked if " " in t]
    kept = [t for t in ranked
            if " " in t or not any(t in b.split() for b in bigrams)]
    return kept[:top_n]


def keyword_score(resume_text, jd_text, top_n=20):
    keywords = extract_keywords(jd_text, top_n)
    r = resume_text.lower()
    matched, missing = [], []
    for k in keywords:
        if re.search(r"\b" + re.escape(k) + r"\b", r):
            matched.append(k)
        else:
            missing.append(k)
    return len(matched) / max(len(keywords), 1), matched, missing


def semantic_score(embedder, resume_text, jd_text):
    emb = embedder.encode([resume_text, jd_text], normalize_embeddings=True)
    sim = float(np.dot(emb[0], emb[1]))  # normalised vectors -> cosine similarity
    return max(0.0, min(1.0, sim))


def format_score(layout, sections, contact, ocr_conf):
    score = 100
    if layout["table"]:
        score -= 15
    if layout["columns"]:
        score -= 15
    score -= 8 * sum(not v for k, v in sections.items()
                     if k in ("experience", "education", "skills"))
    score -= 10 * (not contact["email"])
    score -= 5 * (not contact["phone"])
    if ocr_conf < 0.6:
        score -= 15
    return max(score, 0)


def suggestions(res):
    tips = []
    if res["layout"]["table"]:
        tips.append("Remove tables: many ATS parsers read table cells out of order.")
    if res["layout"]["columns"]:
        tips.append("Use a single-column layout so text is read in the right order.")
    for name in ("experience", "education", "skills"):
        if not res["sections"][name]:
            tips.append(f"Add a clear '{name.title()}' heading.")
    if not res["contact"]["email"]:
        tips.append("Add a plain-text email address.")
    if not res["contact"]["phone"]:
        tips.append("Add a phone number.")
    if res["ocr_conf"] < 0.6:
        tips.append("Text was hard to read: use a larger, standard font "
                    "and a clean, high-contrast file.")
    if res["missing"]:
        tips.append("Add these job keywords if they are true for you: "
                    + ", ".join(res["missing"][:8]) + ".")
    return tips


# ------------------------------------------------------------ main entry
def analyze(data, filename, jd_text, reader, embedder):
    """Run the whole pipeline and return a dict of results."""
    pages = load_pages(data, filename)
    if not pages:
        raise ValueError("Could not read the resume. Upload a PDF, PNG or JPG.")

    texts, confs, images, layout = [], [], [], None
    for page in pages:
        gray = preprocess(page)
        if layout is None:
            layout = layout_report(gray)
        text, conf, results = run_ocr(reader, gray)
        texts.append(text)
        confs.append(conf)
        images.append(annotate(page, results))

    resume_text = " ".join(texts)
    ocr_conf = float(np.mean(confs)) if confs else 0.0
    sections = check_sections(resume_text)
    contact = check_contact(resume_text)
    kw, matched, missing = keyword_score(resume_text, jd_text)
    sem = semantic_score(embedder, resume_text, jd_text)
    fmt = format_score(layout, sections, contact, ocr_conf)
    final = 0.40 * kw * 100 + 0.35 * sem * 100 + 0.25 * fmt

    res = {
        "final": final, "keyword": kw, "semantic": sem, "format": fmt,
        "ocr_conf": ocr_conf, "layout": layout, "sections": sections,
        "contact": contact, "matched": matched, "missing": missing,
        "images": images, "text": resume_text,
    }
    res["tips"] = suggestions(res)
    return res


def report_text(res):
    """Plain-text version of the report (used for the download button)."""
    yn = lambda b: "YES" if b else "NO"
    lines = [
        "=========== AI ATS SYSTEM REPORT ===========",
        f"Generated at: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        f"FINAL ATS SCORE : {res['final']:.0f}/100",
        f"- Keyword match  : {res['keyword']:.0%} ({len(res['matched'])} matched)",
        f"- Semantic match : {res['semantic']:.0%}",
        f"- Format score   : {res['format']}/100",
        f"- OCR confidence : {res['ocr_conf']:.0%}",
        "",
        "Layout analysis (OpenCV):",
        f"  - Table detected : {yn(res['layout']['table'])}",
        f"  - Multi-column   : {yn(res['layout']['columns'])}",
        f"  - Email found    : {yn(res['contact']['email'])}",
        f"  - Phone found    : {yn(res['contact']['phone'])}",
        "",
        "Sections:",
    ]
    for name, ok in res["sections"].items():
        lines.append(f"  - {name}: {'OK' if ok else 'MISSING'}")
    lines += [
        "",
        "Matched keywords: " + (", ".join(res["matched"]) or "None"),
        "Missing keywords: " + (", ".join(res["missing"]) or "None"),
        "",
        "Suggestions:",
    ]
    lines += [f"  * {t}" for t in res["tips"]] or ["  * None"]
    lines.append("=============================================")
    return "\n".join(lines)
