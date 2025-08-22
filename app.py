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

# ---------------- PDF Functions (kept & optimized) ----------------
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

def export_to_pdf(content, filename="output.pdf"):
    """Export content to PDF."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in content.split("\n"):
        pdf.multi_cell(0, 8, line)
    pdf.output(filename)
    return filename

# ---------------- Feedback (kept) ----------------
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

# ---------------- Default Health Reminders (kept) ----------------
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
    st.title("🌍 " + tr("title", "en"))  # title text translated in-tab
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

# Disclaimer & Privacy (translated)
st.warning(f"⚠️ **{tr('disclaimer', lang)}**\n\n🔒 {tr('privacy', lang)}")

username = st.text_input("Enter your name:", "Student")

# ---------------- Tabs (kept original + new) ----------------
tabs = [
    "📄 PDF Q&A",                # kept
    "⏳ Time Management",        # kept
    "💊 Health Reminders",       # kept
    "🥗 Nutrition Assistant",    # kept
    "📊 Dashboard",              # kept
    "💬 Chatbot",                # NEW
    "💊 Medication Tracker",     # NEW
    "🩺 Symptom Checker",        # NEW
]
choice = st.sidebar.radio("Navigate:", tabs)

# Persist last PDF text for chatbot context (optional)
if "last_pdf_text" not in st.session_state:
    st.session_state.last_pdf_text = ""

# ---------------- Tab 1: PDF Q&A (Streaming) ----------------
if choice == "📄 PDF Q&A":
    st.header("Upload a PDF and Ask Questions")
    st.info("📄 Upload your medical PDF and ask a question. You can download the answer as a PDF.")
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    user_query = st.text_input("Enter your question:")
    answer = ""

    if pdf_file and user_query:
        with st.spinner("Extracting and analyzing..."):
            pdf_text = extract_text_from_pdf(pdf_file)
            st.session_state.last_pdf_text = pdf_text  # save for chatbot
            try:
                st.subheader("Answer")
                prompt = f"PDF content:\n{pdf_text}\nQuestion: {user_query}"
                answer = stream_gemini_response(prompt)
                if not answer.strip():
                    answer = summarize_with_gemini(pdf_text, user_query)
            except Exception:
                st.error("⚠️ Could not fetch AI response. Showing quick health tip instead.")
                answer = "Quick Tip: Always take TB meds on time, hydrate well, and rest properly."
                st.write(answer)

        if st.button("Download Answer as PDF"):
            filename = export_to_pdf(answer, "answer.pdf")
            with open(filename, "rb") as f:
                st.download_button("Download PDF", f, file_name="answer.pdf")

        # Feedback
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

# ---------------- Tab 2: Time Management (kept) ----------------
elif choice == "⏳ Time Management":
    st.header("Set Study & Rest Schedule")
    st.info("⏰ Use this to plan your daily study and rest hours.")
    study_hours = st.slider("Study hours per day:", 1, 12, 4)
    rest_hours = 24 - study_hours
    st.success(f"✅ You should rest for {rest_hours} hours per day.")
    st.info("💡 Tip: Balance study, rest, and health care for better focus!")

# ---------------- Tab 3: Health Reminders (kept) ----------------
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

# ---------------- Tab 4: Nutrition Assistant (Streaming; kept) ----------------
elif choice == "🥗 Nutrition Assistant":
    st.header("AI-Powered Nutrition Assistant")
    st.info("🥗 Enter your dietary preferences or health goal and get a personalized meal plan.")
    user_input = st.text_area("Dietary preferences or health goal:")
    nutrition_text = ""

    if st.button("Get Nutrition Advice"):
        if user_input.strip() == "":
            st.warning("Please enter your preferences or health goal.")
        else:
            with st.spinner("Generating personalized advice..."):
                try:
                    prompt = f"""
                    You are a health assistant specialized in nutrition for TB patients.
                    Generate a simple daily meal plan and explain why each food is beneficial.
                    Keep it concise and student-friendly.
                    User input: {user_input}
                    """
                    st.success("Here’s your personalized nutrition advice:")
                    nutrition_text = stream_gemini_response(prompt)
                    if not nutrition_text.strip():
                        nutrition_text = summarize_with_gemini("Nutrition guidance for TB recovery.", user_input)
                        st.write(nutrition_text)
                except Exception:
                    st.error("⚠️ Could not fetch AI advice. Here’s a quick nutrition tip:")
                    nutrition_text = "Eat balanced meals with lean protein, fruits, vegetables, and whole grains. Hydrate well. Avoid processed foods."

    if nutrition_text:
        if st.button("Download Meal Plan as PDF"):
            filename = export_to_pdf(nutrition_text, "nutrition_plan.pdf")
            with open(filename, "rb") as f:
                st.download_button("Download PDF", f, file_name="nutrition_plan.pdf")

        # Feedback
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

# ---------------- Tab 5: Dashboard (extended) ----------------
elif choice == "📊 Dashboard":
    st.header("Your Progress Dashboard")
    st.info("📊 Feedback and medication adherence at a glance.")

    # Feedback chart (kept)
    feedback_list = load_feedback()
    if feedback_list:
        thumbs_up, thumbs_down = get_feedback_stats(feedback_list)
        st.subheader("👍 AI Feedback Summary")
        st.write(f"Total Feedback Entries: {len(feedback_list)}")
        st.write(f"Positive: {thumbs_up}, Negative: {thumbs_down}")
        st.bar_chart({"👍 Yes": [thumbs_up], "👎 No": [thumbs_down]})
    else:
        st.info("No feedback data yet.")

    # NEW: Adherence chart
    st.subheader("💊 Medication Adherence (last 7 days)")
    adh = adherence_series(username, days=7)
    st.line_chart(adh)

# ---------------- Tab 6: Chatbot (NEW) ----------------
elif choice == "💬 Chatbot":
    st.header("Chatbot (fast streaming)")
    use_pdf_context = st.checkbox("Use last uploaded PDF as context (if available)")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_msg = st.text_input("Message:")
    if st.button("Send") and user_msg.strip():
        # Render chat history
        for role, msg in st.session_state.chat_history:
            st.chat_message(role).markdown(msg)

        # Build prompt
        context = st.session_state.last_pdf_text if (use_pdf_context and st.session_state.last_pdf_text) else ""
        prompt = f"Context:\n{context}\n\nUser: {user_msg}\nAssistant:"

        st.chat_message("user").markdown(user_msg)
        with st.chat_message("assistant"):
            try:
                reply = stream_gemini_response(prompt)
                if not reply.strip():
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    reply = model.generate_content(prompt).text
                st.session_state.chat_history.append(("user", user_msg))
                st.session_state.chat_history.append(("assistant", reply))
            except Exception:
                fallback = "⚠️ AI unavailable. Try again soon."
                st.write(fallback)
                st.session_state.chat_history.append(("user", user_msg))
                st.session_state.chat_history.append(("assistant", fallback))

    elif st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.success("Chat cleared.")

# ---------------- Tab 7: Medication Tracker (NEW) ----------------
elif choice == "💊 Medication Tracker":
    st.header("Track your meds & mark doses")
    payload = load_meds(username)
    meds = payload.get("meds", [])

    # Add med
    with st.form("add_med_form"):
        st.subheader("Add / Update a medication")
        med_name = st.text_input("Medication name")
        dose = st.text_input("Dose (e.g., 300mg)")
        time_str = st.text_input("Usual time (e.g., 08:00)")
        submit_med = st.form_submit_button("Save Medication")
    if submit_med and med_name.strip():
        # Upsert
        existing = [m for m in meds if m["name"].lower() == med_name.lower()]
        if existing:
            existing[0]["dose"] = dose
            existing[0]["time"] = time_str
        else:
            meds.append({"name": med_name, "dose": dose, "time": time_str})
        payload["meds"] = meds
        save_meds(username, payload)
        st.success("Medication saved.")

    # List + mark taken
    if meds:
        st.subheader("Today’s meds")
        for m in meds:
            cols = st.columns([3,1])
            with cols[0]:
                st.write(f"- **{m['name']}** — {m.get('dose','')} at {m.get('time','')}")
            with cols[1]:
                if st.button(f"Mark taken ✅ ({m['name']})"):
                    mark_taken(username, m["name"])
                    st.success(f"Marked {m['name']} as taken.")
    else:
        st.info("No medications added yet.")

    # adherence chart also here
    st.subheader("Adherence (last 7 days)")
    st.line_chart(adherence_series(username))

# ---------------- Tab 8: Symptom Checker (NEW) ----------------
elif choice == "🩺 Symptom Checker":
    st.header("Non-diagnostic symptom checker")
    st.info("This is not medical advice. Severe symptoms → seek urgent care.")

    cough = st.selectbox("Cough duration", ["None", "<2 weeks", "≥2 weeks"])
    fever = st.selectbox("Fever", ["No", "Mild", "High"])
    weight_loss = st.checkbox("Unintentional weight loss")
    night_sweats = st.checkbox("Night sweats")
    fatigue = st.checkbox("Fatigue")

    if st.button("Assess"):
        signals = []
        if cough == "≥2 weeks": signals.append("persistent cough")
        if fever in ["Mild", "High"]: signals.append("fever")
        if weight_loss: signals.append("weight loss")
        if night_sweats: signals.append("night sweats")
        if fatigue: signals.append("fatigue")

        if not signals:
            summary = "Low concern from provided info. Monitor and maintain healthy habits."
        else:
            summary = (
                f"Noted: {', '.join(signals)}. "
                "Consider contacting a healthcare professional for guidance, especially if symptoms persist or worsen."
            )
        st.write(summary)

        # Offer AI wording help (fast, concise)
        try:
            prompt = f"Rewrite clearly for a student: {summary}"
            ai_text = stream_gemini_response(prompt)
        except Exception:
            ai_text = summary

        # Export
        if st.button("Download summary PDF"):
            fname = export_to_pdf(ai_text, "symptom_summary.pdf")
            with open(fname, "rb") as f:
                st.download_button("Download PDF", f, file_name="symptom_summary.pdf")

        st.caption("You can share this summary with a clinician.")

# ---------------- Footer ----------------
st.caption("AI-generated insights; always consult a doctor for medical advice.")

