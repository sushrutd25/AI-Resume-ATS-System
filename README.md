# AI Resume ATS System

An automated resume screening tool that uses **machine vision (OpenCV)** and **deep learning (EasyOCR + Sentence Transformers)** to read a resume the way an Applicant Tracking System does, then scores it against a job description.

Unlike plain text parsers, it treats the resume as an **image**. That lets it handle scanned resumes and detect layouts (tables, multiple columns) that real ATS software often fails on.

> Built for the *Machine Vision System* course (BE), Experiment 8.

---

## Features

- **Reads PDFs and images**: pages are rasterised at 200 DPI with PyMuPDF.
- **OpenCV preprocessing**: grayscale, denoising and deskewing for cleaner OCR.
- **Layout audit (OpenCV)**: detects tables and multi-column layouts that hurt ATS parsing.
- **Deep learning OCR**: EasyOCR (ResNet + BiLSTM) extracts text and a confidence score for each word.
- **Annotated output image**: green boxes show confidently read text, red boxes show low-confidence text.
- **Hybrid matching**: TF-IDF keyword matching plus semantic matching with `all-MiniLM-L6-v2` embeddings.
- **Section and contact checks**: looks for Experience, Education, Skills, Projects, email and phone.
- **Live monitoring**: the program watches your files and refreshes the report whenever the resume or job description changes.

## How it works

```
Resume (PDF/image)
      |
      v
 1. Rasterise page to pixel matrix      (PyMuPDF)
 2. Denoise + deskew                     (OpenCV)
 3. Layout analysis: tables, columns     (OpenCV morphology)
 4. Text extraction + confidence         (EasyOCR)
 5. Keyword match (TF-IDF) + semantic match (Sentence Transformers)
      |
      v
 ATS score + annotated image + live text report
```

### Scoring

| Component | Weight | What it measures |
| --- | --- | --- |
| Keyword match | 40% | How many top job-description terms appear in the resume |
| Semantic match | 35% | Meaning-level similarity between resume and job description |
| Format score | 25% | Layout risks, missing sections, missing contact info, OCR confidence |

`Final score = 0.40 x keyword + 0.35 x semantic + 0.25 x format`

## Tech stack

Python 3.11, OpenCV, EasyOCR, Sentence-Transformers, scikit-learn, PyMuPDF, NumPy.

## Project structure

```
.
├── main.py             # the ATS pipeline
├── requirements.txt    # dependencies
├── samples/            # put resume.pdf and job.txt here
├── output/             # generated: annotated images and ats_report.txt
├── docs/images/        # screenshots used in this README
├── LICENSE
└── README.md
```

## Installation (Windows)

Python **3.11** is recommended. Newer versions may not have PyTorch builds yet.

```powershell
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

On macOS or Linux, activate with `source venv/bin/activate` instead.

> If PowerShell blocks the activate script, run
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once.

## Usage

1. Put a resume at `samples/resume.pdf` (a PDF or image). Use a sample resume, not one with personal details you don't want shared.
2. Put the job description as plain text at `samples/job.txt` (a PDF also works).
3. Run:

```powershell
python main.py
```

The first run downloads the OCR and embedding models, which can take a few minutes. After that the program keeps running and refreshes `output/ats_report.txt` automatically whenever `samples/resume.pdf` or `samples/job.txt` is saved. Press `Ctrl + C` to stop.


## Limitations

- Keyword extraction is purely statistical, so generic words can still slip through.
- Layout checks are rule-based (OpenCV), not a trained detector.
- The scoring weights are a reasonable starting point, not calibrated against real ATS software.
- OCR on very small or heavily decorated fonts can be inaccurate.

## Roadmap

- [ ] Streamlit web interface for upload-and-score
- [ ] Section-by-section matching (skills vs. required skills)
- [ ] Batch ranking of many resumes against one job description
- [ ] Trained layout detection model

## Team

| Name | Roll No. |

| Sushrut Diwan | 23RA015 |

## License

Released under the [MIT License](LICENSE).
