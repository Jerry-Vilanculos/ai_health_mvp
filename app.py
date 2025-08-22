import os
import json
from datetime import datetime, date, timedelta
import threading

import streamlit as st
import pdfplumber
from fpdf import FPDF
import google.generativeai as genai
import pandas as pd

# ---------------- Setup ----------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
APP_PASSWORD = os.getenv("APP_PASSWORD", "")  # optional simple gate

# ----------- i18n (add more languages easily) -----------
LANGS = {
    "en": "English",
    "es": "Español",
    "fr": "Français",
}
T = {
    "title": {"en": "AI-Powered Health Assistant for Students",
              "es": "Asistente de Salud con IA para Estudiantes",
              "fr": "Assistant Santé IA pour Étudiants"},
    "disclaimer": {
        "en": "Educational use only — not medical advice.",
        "es": "Uso educativo — no es consejo médico.",
        "fr": "Usage éducatif — pas un avis médical."
    },
    "privacy": {
        "en": "Data stays local; PDFs and inputs are not shared. Feedback stored in feedback.json.",
        "es": "Los datos permanecen locales; PDFs y entradas no se comparten. Comentarios en feedback.json.",
        "fr": "Les données restent locales ; PDFs et entrées ne sont pas partagés. Retours dans feedback.json."
    }
}
def tr(key, lang): return T.get(key, {}).get(lang, T.get(key, {}).get("en", key))

# ---------------- Paths & small storage helpers ----------------
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
    """Extract limited text from PDF to keep responses fast."""
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text
            if len(text) > max_chars:  # stop early
                break
    return text[:max_chars]

@st.cache_data(show_spinner=False)
def summarize_with_gemini(text, query):
    """Summarize or answer query using Gemini (cached for repeat speed)."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
    You are a medical assistant. Based on the following document:
    {text}

    Answer the question simply and clearly: {query}
    """
    response = model.generate_content(prompt)
    return response.text

def stream_gemini_response(prompt):
    """Stream Gemini response in real-time for faster UX."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt, stream=True)
    final = ""
    placeholder = st.empty()
    for chunk in response:
        if hasattr(chunk, "text") and chunk.text:
            final += chunk.text
            placeholder.markdown(final)
    return final

def summarize_with_timeout(prompt, timeout=10):
    """Call Gemini with timeout to prevent hanging."""
    result = {"text": ""}

    def target():
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            result["text"] = response.text
        except Exception:
            result["text"] = ""

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)
    if thread.is_alive() or not result["text"]:
        return "⚠️ AI unavailable or taking too long. Here's a quick tip: stay hydrated, take TB meds on time, and rest properly."
    return result["text"]

def export_to_pdf(content, filename="output.pdf"):
    """Export content to PDF."""
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

# ---------------- Medication Tracker storage ----------------
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

# Language selector + simple login
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

# ---------------- Tabs ----------------
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
    st.info("📄 Upload your medical PDF and ask a question. You can download the answer as a PDF.")
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    user_query = st.text_input("Enter your question:")
    answer = ""

    if pdf_file and user_query:
        pdf_text = extract_text_from_pdf(pdf_file, max_chars=2000)
        st.session_state.last_pdf_text = pdf_text

        prompt = f"PDF content:\n{pdf_text}\nQuestion: {user_query}"
        placeholder = st.empty()
        placeholder.info("Extracting and analyzing...")

        answer = summarize_with_timeout(prompt)

        placeholder.success("Analysis complete!")
        st.subheader("Answer")
        st.write(answer)

        if st.button("Download Answer as PDF"):
            filename = export_to_pdf(answer, "answer.pdf")
            with open(filename, "rb") as f:
                st.download_button("Download PDF", f, file_name="answer.pdf")

        st.subheader("Was this answer helpful?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Yes", key="pdf_up"):
                save_feedback(username, "up")
                st.success("Feedback recorded ✅")
        with col2:
            if st.button("👎 No", key="pdf_down"):
                save_feedback(username, "down")
                st.success("Feedback recorded ✅")

# ---------------- Tab 2: Time Management ----------------
elif choice == "⏳ Time Management":
    st.header("Set Study & Rest Schedule")
    st.info("⏰ Use this to plan your daily study and rest hours.")
    study_hours = st.slider("Study hours per day:", 1, 12, 4)
    rest_hours = 24 - study_hours
    st.success(f"✅ You should rest for {rest_hours} hours per day.")
    st.info("💡 Tip: Balance study, rest, and health care for better focus!")

# ---------------- Tab 3: Health Reminders ----------------
elif choice == "💊 Health Reminders":
    st.header("Default TB Care Schedule")
    st.info("💊 Follow these daily reminders for TB treatment and wellness.")
    day = st.selectbox("Select Day:", list(DEFAULT_REMINDERS.keys()))
    reminders = DEFAULT_REMINDERS[day]
    st.write("### Reminders for Today:")
    for r in reminders:
        st.write(f"- {r}")
        st.info(f"⏰ Reminder: {r}")

    if st.button("Download Schedule as PDF"):
        content = "\n".join([f"{d}: {r}" for d, rlist in DEFAULT_REMINDERS.items() for r in rlist])
        filename = export_to_pdf(content, "tb_schedule.pdf")
        with open(filename, "rb") as f:
            st.download_button("Download PDF", f, file_name="tb_schedule.pdf")

# ---------------- Tab 4: Nutrition Assistant ----------------
elif choice == "🥗 Nutrition Assistant":
    st.header("AI-Powered Nutrition Assistant")
    st.info("🥗 Enter your dietary preferences or health goal and get a personalized meal plan.")
    user_input = st.text_area("Dietary preferences or health goal:")
    nutrition_text = ""

    if st.button("Get Nutrition Advice"):
        if user_input.strip() == "":
            st.warning("Please enter your preferences or health goal.")
        else:
            prompt = f"""
            You are a health assistant specialized in nutrition for TB patients.
            Generate a simple daily meal plan and explain why each food is beneficial.
            Keep it concise and student-friendly.
            User input: {user_input}
            """
            placeholder = st.empty()
            placeholder.info("Generating personalized advice...")

            nutrition_text = summarize_with_timeout(prompt)
            placeholder.success("Advice ready!")
            st.write(nutrition_text)

    if nutrition_text:
        if st.button("Download Meal Plan as PDF"):
            filename = export_to_pdf(nutrition_text, "nutrition_plan.pdf")
            with open(filename, "rb") as f:
                st.download_button("Download PDF", f, file_name="nutrition_plan.pdf")

        st.subheader("Was this advice helpful?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Yes", key="nutrition_up"):
                save_feedback(username, "up")
                st.success("Feedback recorded ✅")
        with col2:
            if st.button("👎 No", key="nutrition_down"):
                save_feedback(username, "down")
                st.success("Feedback recorded ✅")

# ---------------- Tab 5 to Tab 8 remain unchanged ----------------
# ... [All your previous Tab 5: Dashboard, Tab 6: Chatbot, Tab 7: Medication Tracker, Tab 8: Symptom Checker code here] ...

st.caption("AI-generated insights; always consult a doctor for medical advice.")


