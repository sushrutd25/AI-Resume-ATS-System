"""
Streamlit web app for the AI Resume ATS.
Run with:  streamlit run app.py
"""
import cv2
import streamlit as st

import ats_core as core

st.set_page_config(page_title="AI Resume ATS", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading AI models (first time only, please wait)...")
def get_models():
    """Load the heavy models once and reuse them on every click."""
    return core.load_reader(), core.load_embedder()


# ---------------------------------------------------------------- header
st.title("📄 AI Resume ATS System")
st.caption("Machine vision (OpenCV) + deep learning (EasyOCR, Sentence Transformers). "
           "Upload a resume, paste a job description, and get an ATS score.")

# ---------------------------------------------------------------- inputs
left, right = st.columns(2)
with left:
    resume = st.file_uploader("1. Upload your resume", type=["pdf", "png", "jpg", "jpeg"])
with right:
    jd_text = st.text_area("2. Paste the job description", height=200,
                           placeholder="Paste the full job posting here...")

ready = resume is not None and len(jd_text.strip()) > 30
if st.button("Analyze resume", type="primary", disabled=not ready):
    reader, embedder = get_models()
    with st.spinner("Reading the resume and scoring it... this can take up to a minute."):
        try:
            st.session_state["result"] = core.analyze(
                resume.getvalue(), resume.name, jd_text, reader, embedder)
        except Exception as err:
            st.session_state.pop("result", None)
            st.error(f"Something went wrong: {err}")

if not ready:
    st.info("Upload a resume and paste a job description (a few lines at least) "
            "to enable the Analyze button.")

# ---------------------------------------------------------------- results
res = st.session_state.get("result")
if res:
    st.divider()
    final = res["final"]
    label = "Strong match" if final >= 75 else "Fair match" if final >= 50 else "Needs work"

    st.subheader(f"Final ATS score: {final:.0f} / 100  ({label})")
    st.progress(min(int(final), 100))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Keyword match", f"{res['keyword']:.0%}")
    c2.metric("Semantic match", f"{res['semantic']:.0%}")
    c3.metric("Format score", f"{res['format']}/100")
    c4.metric("OCR confidence", f"{res['ocr_conf']:.0%}")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Keywords", "Layout and sections", "Annotated resume", "Suggestions"])

    with tab1:
        a, b = st.columns(2)
        a.markdown("**Matched keywords**")
        a.write(", ".join(res["matched"]) or "None")
        b.markdown("**Missing keywords** (add only if true for you)")
        b.write(", ".join(res["missing"]) or "None")

    with tab2:
        a, b = st.columns(2)
        a.markdown("**Layout check (OpenCV)**")
        a.write(f"{'⚠️' if res['layout']['table'] else '✅'} Table detected: "
                f"{'yes' if res['layout']['table'] else 'no'}")
        a.write(f"{'⚠️' if res['layout']['columns'] else '✅'} Multi-column layout: "
                f"{'yes' if res['layout']['columns'] else 'no'}")
        a.write(f"{'✅' if res['contact']['email'] else '⚠️'} Email found: "
                f"{'yes' if res['contact']['email'] else 'no'}")
        a.write(f"{'✅' if res['contact']['phone'] else '⚠️'} Phone found: "
                f"{'yes' if res['contact']['phone'] else 'no'}")
        b.markdown("**Sections found**")
        for name, ok in res["sections"].items():
            b.write(f"{'✅' if ok else '⚠️'} {name.title()}")

    with tab3:
        st.caption("Green boxes = text read confidently. Red boxes = low-confidence text.")
        for i, img in enumerate(res["images"], 1):
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Page {i}",
                     use_container_width=True)

    with tab4:
        if res["tips"]:
            for tip in res["tips"]:
                st.write(f"- {tip}")
        else:
            st.success("No major issues found.")

    with st.expander("Text extracted by OCR"):
        st.text(res["text"])

    st.download_button("Download report (.txt)", core.report_text(res),
                       file_name="ats_report.txt")
