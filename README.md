# AI-GAME
# 📄 ATS Resume Checker

Upload a resume (PDF, DOCX or TXT) and get an estimated **ATS score**, a score breakdown,
prioritized **improvements**, and missing keywords. Optionally paste a job description for
better keyword matching. Built with **Streamlit** and **Google Gemini Flash**.

> No real ATS publishes a score. The number is an AI estimate of parsing-friendliness and
> keyword/impact quality. Treat it as guidance, not a guarantee.

## Run locally

```bash
git clone https://github.com/<your-username>/ats-resume-checker.git
cd ats-resume-checker
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Get a free API key at https://aistudio.google.com/apikey, then either:

- paste it in the app sidebar, or
- create `.streamlit/secrets.toml` (never commit this file):

```toml
GEMINI_API_KEY = "your-key-here"
# GEMINI_MODEL = "gemini-3.5-flash"   # optional, change if Google renames the model
```

Run:

```bash
streamlit run app.py
```

## Push to GitHub

```bash
git init
git add app.py requirements.txt README.md .gitignore
git commit -m "Initial commit: ATS resume checker"
git branch -M main
# create an empty repo on github.com first, then:
git remote add origin https://github.com/<your-username>/ats-resume-checker.git
git push -u origin main
```

## Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app** → choose your repo, branch `main`, main file `app.py`.
3. Open **Advanced settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
4. Click **Deploy**. You get a public `*.streamlit.app` URL.

## Notes

- Scanned/image-only PDFs have no text to read; export a text-based PDF or use DOCX.
- Default model is `gemini-3.5-flash`. If you get a "model not found" error, set
  `GEMINI_MODEL` (or edit the Model box in the sidebar) to a current Flash model name.
- Resume text is sent to the Gemini API. Don't upload documents you can't share with it.
