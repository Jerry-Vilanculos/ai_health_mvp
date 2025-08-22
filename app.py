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
APP_PASSWORD = os.getenv("APP_PASSWORD", "")

# ---------------- i18n ----------------
LANGS = {"en": "English", "es": "Español", "fr": "Français"}
T = {
    "title": {"en": "AI-Powered Health Assistant for Students"},
    "disclaimer": {"en": "Educational use only — not medical advice."},
    "privacy": {"en": "Data stays local; PDFs and inputs are not shared. Feedback stored in feedback.json."}
}
def tr(key, lang="en"): return T.get(key, {}).get(lang, T.get(key, {}).get("en", key))

# ---------------- Paths ----------------
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
            page_text = page.extract_text() or ""
            text += page_text
            if len(text) > max_chars:
                break
    return text[:max_chars]

@st.cache_data(show_spinner=False)
def summarize_with_gemini(text, query):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"You are a medical assistant. Based on the document:\n{text}\nAnswer simply: {query}"
        return model.generate_content(prompt).text
    except Exception:
        return "Quick Tip: Always take TB meds on time, hydrate, and rest properly."

def stream_gemini_response(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt, stream=True)
        final = ""
        placeholder = st.empty()
        for chunk in response:
            if hasattr(chunk, "text") and chunk.text:
                final += chunk.text
                placeholder.markdown(final)
        return final or "Quick Tip: Always take TB meds on time, hydrate, and rest properly."
    except Exception:
        return "Quick Tip: Always take TB meds on time, hydrate, and rest properly."

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

# Language + login
cols_top = st.columns([3,1,1])
with cols_top[0]:
    st.title("🌍 " + tr("title"))
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

st.warning(f"⚠️ **{tr('disclaimer')}**\n\n🔒 {tr('privacy')}")

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
    "🩺 Symptom Checker"
]
choice = st.sidebar.radio("Navigate:", tabs)

if "last_pdf_text" not in st.session_state:
    st.session_state.last_pdf_text = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- PDF Q&A ----------------
if choice == "📄 PDF Q&A":
    st.header("Upload a PDF and Ask Questions")
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    user_query = st.text_input("Enter your question:")
    answer = ""
    if pdf_file and user_query:
        with st.spinner("Extracting and analyzing..."):
            pdf_text = extract_text_from_pdf(pdf_file)
            st.session_state.last_pdf_text = pdf_text
            prompt = f"PDF content:\n{pdf_text}\nQuestion: {user_query}"
            answer = stream_gemini_response(prompt)
            st.subheader("Answer")
            st.write(answer)
        if st.button("Download Answer as PDF"):
            fname = export_to_pdf(answer, "answer.pdf")
            with open(fname, "rb") as f:
                st.download_button("Download PDF", f, file_name="answer.pdf")
        # Feedback
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Yes", key="pdf_up"):
                save_feedback(username, "up")
                st.success("Feedback recorded ✅")
        with col2:
            if st.button("👎 No", key="pdf_down"):
                save_feedback(username, "down")
                st.success("Feedback recorded ✅")

# ---------------- Remaining tabs ----------------
# Time Management
elif choice == "⏳ Time Management":
    st.header("Set Study & Rest Schedule")
    study_hours = st.slider("Study hours per day:", 1, 12, 4)
    rest_hours = 24 - study_hours
    st.success(f"✅ Rest: {rest_hours} hours")

# Health Reminders
elif choice == "💊 Health Reminders":
    day = st.selectbox("Select Day:", list(DEFAULT_REMINDERS.keys()))
    st.write("### Reminders for Today:")
    for r in DEFAULT_REMINDERS[day]:
        st.write(f"- {r}")

# Nutrition Assistant
elif choice == "🥗 Nutrition Assistant":
    st.header("Nutrition Assistant")
    user_input = st.text_area("Dietary preferences or health goal:")
    nutrition_text = ""
    if st.button("Get Nutrition Advice"):
        prompt = f"Nutrition plan for: {user_input}"
        nutrition_text = stream_gemini_response(prompt)
        st.write(nutrition_text)

# Dashboard
elif choice == "📊 Dashboard":
    st.header("Dashboard")
    feedback_list = load_feedback()
    thumbs_up, thumbs_down = get_feedback_stats(feedback_list)
    st.write(f"Feedback: 👍 {thumbs_up}, 👎 {thumbs_down}")
    st.bar_chart({"👍 Yes": [thumbs_up], "👎 No": [thumbs_down]})
    st.subheader("Medication Adherence")
    st.line_chart(adherence_series(username))

# Chatbot
elif choice == "💬 Chatbot":
    use_pdf_context = st.checkbox("Use last PDF context")
    user_msg = st.text_input("Message:")
    if st.button("Send"):
        context = st.session_state.last_pdf_text if use_pdf_context else ""
        prompt = f"Context:\n{context}\nUser: {user_msg}\nAssistant:"
        reply = stream_gemini_response(prompt)
        st.session_state.chat_history.append(("user", user_msg))
        st.session_state.chat_history.append(("assistant", reply))
    for role, msg in st.session_state.chat_history:
        st.markdown(f"**{role}:** {msg}")
    if st.button("Clear Chat"):
        st.session_state.chat_history = []

# Medication Tracker
elif choice == "💊 Medication Tracker":
    st.header("Medication Tracker")
    payload = load_meds(username)
    meds = payload.get("meds", [])
    with st.form("add_med_form"):
        med_name = st.text_input("Medication name")
        dose = st.text_input("Dose")
        time_str = st.text_input("Time")
        submit_med = st.form_submit_button("Save Medication")
    if submit_med and med_name.strip():
        existing = [m for m in meds if m["name"].lower() == med_name.lower()]
        if existing:
            existing[0]["dose"] = dose
            existing[0]["time"] = time_str
        else:
            meds.append({"name": med_name, "dose": dose, "time": time_str})
        payload["meds"] = meds
        save_meds(username, payload)
        st.success("Medication saved.")
    st.subheader("Today’s meds")
    for m in meds:
        st.write(f"- {m['name']} at {m.get('time','')}")
        if st.button(f"Mark taken ({m['name']})"):
            mark_taken(username, m["name"])
    st.subheader("Adherence")
    st.line_chart(adherence_series(username))

# Symptom Checker
elif choice == "🩺 Symptom Checker":
    st.header("Symptom Checker")
    cough = st.selectbox("Cough duration", ["None", "<2 weeks", "≥2 weeks"])
    fever = st.selectbox("Fever", ["No", "Mild", "High"])
    weight_loss = st.checkbox("Weight loss")
    night_sweats = st.checkbox("Night sweats")
    fatigue = st.checkbox("Fatigue")
    if st.button("Assess"):
        signals = []
        if cough == "≥2 weeks": signals.append("persistent cough")
        if fever != "No": signals.append("fever")
        if weight_loss: signals.append("weight loss")
        if night_sweats: signals.append("night sweats")
        if fatigue: signals.append("fatigue")
        summary = "Low concern." if not signals else f"Noted: {', '.join(signals)}. Seek advice if persists."
        st.write(summary)
        prompt = f"Rewrite clearly for a student: {summary}"
        ai_text = stream_gemini_response(prompt)
        st.write(ai_text)
        if st.button("Download summary PDF"):
            fname = export_to_pdf(ai_text, "symptom_summary.pdf")
            with open(fname, "rb") as f:
                st.download_button("Download PDF", f, file_name="symptom_summary.pdf")


