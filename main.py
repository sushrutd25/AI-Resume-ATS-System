"""
AI Resume ATS System (Machine Vision + Deep Learning)

Pipeline:
  Resume (PDF/image) -> OpenCV preprocessing -> OpenCV layout checks
  -> EasyOCR (deep learning text recognition) -> section/contact checks
  -> keyword match (TF-IDF) + semantic match (Sentence-Transformers)
  -> final ATS score + annotated image

Usage:
  python main.py --resume samples/resume.pdf --jd samples/job.txt
"""
import argparse
import re
import sys
from pathlib import Path

import cv2
import fitz  # PyMuPDF: renders PDF pages to images
import numpy as np
import easyocr
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer

OUTPUT_DIR = Path("output")
SECTIONS = {
    "summary": r"\b(summary|objective|profile)\b",
    "experience": r"\b(experience|employment|work history|internship)\b",
    "education": r"\b(education|academic|qualification)\b",
    "skills": r"\b(skills|technologies|technical skills)\b",
    "projects": r"\b(projects?)\b",
}


# ---------------------------------------------------------------- 1. LOAD
def load_pages(path: str):
    """Return a list of BGR images (one per page)."""
    p = Path(path)
    if not p.exists():
        sys.exit(f"File not found: {path}")
    if p.suffix.lower() == ".pdf":
        pages = []
        for page in fitz.open(p):
            pix = page.get_pixmap(dpi=200, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8)
            img = img.reshape(pix.height, pix.width, 3)
            pages.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return pages
    img = cv2.imread(str(p))
    if img is None:
        sys.exit("Could not read image.")
    return [img]


# ------------------------------------------------------------ 2. PREPROCESS
def preprocess(img):
    """Grayscale -> denoise -> deskew (helps OCR on scanned resumes)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, h=10)

    inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(inv > 0))
    if len(coords) > 100:
        angle = cv2.minAreaRect(coords)[-1]
        if angle > 45:
            angle -= 90
        if 0.5 < abs(angle) < 15:  # only fix small skews
            h, w = gray.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
    return gray


# --------------------------------------------------------- 3. LAYOUT CHECKS
def layout_report(gray):
    """Use OpenCV to find layout features that confuse real ATS parsers."""
    h, w = gray.shape
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 15, 10)

    # Table detection: long horizontal AND vertical lines
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 15, 1), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 15, 1)))
    hl = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk)
    vl = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk)
    n_h = len(cv2.findContours(hl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
    n_v = len(cv2.findContours(vl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
    has_table = n_h >= 2 and n_v >= 2

    # Multi-column detection: an empty vertical gap in the middle of the page
    profile = (bw > 0).sum(axis=0)
    lo, hi = int(w * 0.30), int(w * 0.70)
    empty = profile[lo:hi] <= 0.005 * h
    longest = run = 0
    for e in empty:
        run = run + 1 if e else 0
        longest = max(longest, run)
    has_columns = longest >= w * 0.02

    return {"table": has_table, "columns": has_columns}


# ------------------------------------------------------------------ 4. OCR
def run_ocr(reader, gray):
    """Deep-learning OCR. Returns (text, avg_confidence, boxes)."""
    results = reader.readtext(gray)  # [(bbox, text, conf), ...]
    text = " ".join(r[1] for r in results)
    conf = float(np.mean([r[2] for r in results])) if results else 0.0
    return text, conf, results


def annotate(img, results, name):
    out = img.copy()
    for bbox, _, conf in results:
        pts = np.array(bbox, dtype=np.int32)
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        color = (0, 180, 0) if conf > 0.6 else (0, 0, 255)  # red = low confidence
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / name
    cv2.imwrite(str(path), out)
    return path


# ----------------------------------------------------------- 5. TEXT CHECKS
def check_sections(text):
    t = text.lower()
    return {s: bool(re.search(p, t)) for s, p in SECTIONS.items()}


def check_contact(text):
    return {
        "email": bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)),
        "phone": bool(re.search(r"(\+?\d[\d\s\-()]{8,}\d)", text)),
    }


# ------------------------------------------------------------- 6. SCORING
def keyword_score(resume_text, jd_text, top_n=25):
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                            max_features=top_n)
    tfidf.fit([jd_text])
    keywords = list(tfidf.get_feature_names_out())
    r = resume_text.lower()
    matched = [k for k in keywords if k in r]
    missing = [k for k in keywords if k not in r]
    return len(matched) / max(len(keywords), 1), matched, missing


def semantic_score(resume_text, jd_text):
    model = SentenceTransformer("all-MiniLM-L6-v2")  # downloads once (~90 MB)
    emb = model.encode([resume_text, jd_text], convert_to_tensor=True)
    sim = float(util.cos_sim(emb[0], emb[1]))
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


# -------------------------------------------------------------------- MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", required=True, help="PDF or image of the resume")
    ap.add_argument("--jd", required=True, help="text file with the job description")
    args = ap.parse_args()

    jd_text = Path(args.jd).read_text(encoding="utf-8")
    pages = load_pages(args.resume)
    print(f"Loaded {len(pages)} page(s). Loading OCR model...")
    reader = easyocr.Reader(["en"], gpu=False)

    all_text, confs, layout = [], [], None
    for i, page in enumerate(pages, 1):
        gray = preprocess(page)
        if layout is None:  # check layout on first page
            layout = layout_report(gray)
        text, conf, results = run_ocr(reader, gray)
        all_text.append(text)
        confs.append(conf)
        saved = annotate(page, results, f"annotated_page{i}.png")
        print(f"  page {i}: OCR confidence {conf:.0%} -> {saved}")

    resume_text = " ".join(all_text)
    ocr_conf = float(np.mean(confs)) if confs else 0.0

    sections = check_sections(resume_text)
    contact = check_contact(resume_text)
    kw, matched, missing = keyword_score(resume_text, jd_text)
    sem = semantic_score(resume_text, jd_text)
    fmt = format_score(layout, sections, contact, ocr_conf)

    final = 0.40 * kw * 100 + 0.35 * sem * 100 + 0.25 * fmt

    print("\n========== ATS REPORT ==========")
    print(f"Keyword match   : {kw:.0%}  ({len(matched)} matched)")
    print(f"Semantic match  : {sem:.0%}  (deep learning embeddings)")
    print(f"Format score    : {fmt}/100")
    print(f"FINAL ATS SCORE : {final:.0f}/100")

    print("\n-- Layout (OpenCV) --")
    print("  Table detected   :", "YES (risky)" if layout["table"] else "no")
    print("  Multi-column     :", "YES (risky)" if layout["columns"] else "no")

    print("\n-- Sections found --")
    for s, ok in sections.items():
        print(f"  {'OK ' if ok else 'MISSING'}  {s}")

    print("\n-- Contact --")
    print("  email:", contact["email"], "| phone:", contact["phone"])

    print("\n-- Missing keywords to consider adding --")
    print("  " + (", ".join(missing[:15]) or "none"))


if __name__ == "__main__":
    main()
