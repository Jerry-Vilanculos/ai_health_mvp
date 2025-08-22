import os
import json
from datetime import datetime, date, timedelta

import streamlit as st
import pdfplumber
from fpdf import FPDF
import google.generativeai as genai
import pandas as pd

# ---------------- Setup ----------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
APP_PASSWORD = os.getenv("APP_PASSWORD", "")  # optional simple gate

# ----------- i18n (add more languages easily) -----------
LANGS = {"en": "English", "es": "Español", "fr": "Français"}
T = {
    "title": {
        "en": "AI-Powered Health Assistant for Students",
        "es": "Asistente de Salud con IA para Estudiantes",
        "fr": "Assistant Santé IA pour Étudiants"
    },
    "disclaimer": {
        "en": "Educational use only — not medical advice.",
        "es": "Uso educativo — no es consejo médico.",
        "fr": "Usage éducatif — pas un avis médical"
    },
    "privacy": {
        "en": "Data stays local; PDFs and inputs are not shared. Feedback stored in feedback.json.",
        "es": "Los datos permanecen locales; PDFs y entradas no se comparten. Comentarios en feedback.json.",
        "fr": "Les données restent locales ; PDFs et entrées ne sont pas partagés. Retours dans feedback.json."
    }
}
def tr(key, lang): return T.get(key, {}).get(lang, T.get(key, {}).get("en", key))

# ---------------- Paths & storage helpers ----------------
DATA_DIR = "."
FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback.json")
MEDS_FILE = os.path.join(DATA_DIR, "meds.json")

def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ---------------- PDF Functions ----------------
def extract_text_from_pdf(pdf_file, max_chars=4000):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
            if len(text) > max_chars:
                break
    return text[:max_chars]

@st.cache_data(show_spinner=False)
def summarize_with_gemini(text, query):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"You are a medical assistant. Based on the following document:\n{text}\nAnswer the question simply and clearly: {query}"
    response = model.generate_content(prompt)
    return response.text

def stream_gemini_response(prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt, stream=True)
    final = ""
    placeholder = st.empty()
    for chunk in response:
        if hasattr(chunk, "text") and chunk.text:
            final += chunk.text
            placeholder.markdown(final)
    return final

def export_to_pdf(content, filename="output.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in content.split("\n"):
        pdf.multi_cell(0, 8, line)
    pdf.output(filename)
    return filename

# ---------------- Feedback ----------------
def save_feedback(user, feedback):
    data = load_json(FEEDBACK_FILE, [])
    data.append({"user": user, "feedback": feedback, "timestamp": datetime.now().isoformat()})
    save_json(FEEDBACK_FILE, data)

def load_feedback():
    return load_json(FEEDBACK_FILE, [])

def get_feedback_stats(feedback_list):
    thumbs_up = sum(1 for f in feedback_list if f.get("feedback") == "up")
    thumbs_down = sum(1 for f in feedback_list if f.get("feedback") == "down")
    return thumbs_up, thumbs_down

# ---------------- Default Health Reminders ----------------
DEFAULT_REMINDERS = {
    "Monday": ["Take TB medication at 8 AM", "Drink 2L water"],
    "Tuesday": ["Take TB medication at 8 AM", "Light exercise 20 min"],
    "Wednesday": ["Take TB medication at 8 AM", "Eat protein-rich meal"],
    "Thursday": ["Take TB medication at 8 AM", "Vitamin check"],
    "Friday": ["Take TB medication at 8 AM", "Hydrate properly"],
    "Saturday": ["Take TB medication at 9 AM", "Rest and recovery"],
    "Sunday": ["Take TB medication at 9 AM", "Prepare healthy meals for week"]
}

# ---------------- Medication Tracker ----------------
def load_meds(username):
    data = load_json(MEDS_FILE, {})
    return data.get(username, {"meds": [], "events": []})

def save_meds(username, payload):
    data = load_json(MEDS_FILE, {})
    data[username] = payload
    save_json(MEDS_FILE, data)

def mark_taken(username, med_name):
    payload = load_meds(username)
    payload["events"].append({"med": med_name, "taken_at": datetime.now().isoformat()})
    save_meds(username, payload)

def adherence_series(username, days=7):
    payload = load_meds(username)
    df = pd.DataFrame(payload.get("events", []))
    if df.empty:
        dates = [date.today() - timedelta(days=i) for i in range(days)][::-1]
        return pd.DataFrame({"date": dates, "taken": [0]*days}).set_index("date")
    df["date"] = pd.to_datetime(df["taken_at"]).dt.date
    counts = df.groupby("date").size()
    dates = [date.today() - timedelta(days=i) for i in range(days)][::-1]
    series = [int(counts.get(d, 0)) for d in dates]
    return pd.DataFrame({"date": dates, "taken": series}).set_index("date")

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="AI Health MVP", layout="wide")

cols_top = st.columns([3,1,1])
with cols_top[0]:
    st.title("🌍 " + tr("title", "en"))
with cols_top[1]:
    lang = st.selectbox("Language", options=list(LANGS.keys()), format_func=lambda k: LANGS[k], index=0)
with cols_top[2]:
    if APP_PASSWORD:
        pwd = st.text_input("Passcode", type="password")
        if "authed" not in st.session_state:
            st.session_state.authed = False
        if st.button("Unlock"):
            st.session_state.authed = (pwd == APP_PASSWORD)
        if not st.session_state.authed:
            st.stop()

st.warning(f"⚠️ **{tr('disclaimer', lang)}**\n\n🔒 {tr('privacy', lang)}")

username = st.text_input("Enter your name:", "Student")

tabs = [
    "📄 PDF Q&A",
    "⏳ Time Management",
    "💊 Health Reminders",
    "🥗 Nutrition Assistant",
    "📊 Dashboard",
    "💬 Chatbot",
    "💊 Medication Tracker",
    "🩺 Symptom Checker",
]
choice = st.sidebar.radio("Navigate:", tabs)

if "last_pdf_text" not in st.session_state:
    st.session_state.last_pdf_text = ""

# ---------------- Tab 1: PDF Q&A ----------------
if choice == "📄 PDF Q&A":
    st.header("Upload a PDF and Ask Questions")
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    user_query = st.text_input("Enter your question:")
    answer = ""
    
    if pdf_file:
        pdf_text = extract_text_from_pdf(pdf_file)
        st.session_state.last_pdf_text = pdf_text

    if pdf_file and user_query:
        if st.button("Ask Question"):
            prompt = f"PDF content:\n{pdf_text}\nQuestion: {user_query}"
            answer = stream_gemini_response(prompt)
            if not answer.strip():
                answer = summarize_with_gemini(pdf_text, user_query)
            st.subheader("Answer")
            st.write(answer)

            if st.button("Download Answer as PDF", key="pdf_download"):
                fname = export_to_pdf(answer, "answer.pdf")
                with open(fname,"rb") as f:
                    st.download_button("Download PDF", f, file_name="answer.pdf")

            # Feedback buttons with unique keys
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Yes", key=f"pdf_up_{user_query}"):
                    save_feedback(username, "up")
                    st.success("Feedback recorded ✅")
            with col2:
                if st.button("👎 No", key=f"pdf_down_{user_query}"):
                    save_feedback(username, "down")
                    st.success("Feedback recorded ✅")

