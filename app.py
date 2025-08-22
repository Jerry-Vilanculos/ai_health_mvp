import os
import streamlit as st
import pdfplumber
from fpdf import FPDF
import google.generativeai as genai
import json
from datetime import datetime

# ---------------- Setup ----------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ---------------- PDF Functions ----------------
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def summarize_with_gemini(text, query):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
    You are a medical assistant. Based on the following document:
    {text}

    Answer the question simply and clearly: {query}
    """
    response = model.generate_content(prompt)
    return response.text

def export_to_pdf(content, filename="output.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in content.split("\n"):
        pdf.multi_cell(0, 10, line)
    pdf.output(filename)
    return filename

# ---------------- Feedback Functions ----------------
FEEDBACK_FILE = "feedback.json"

def save_feedback(user, feedback):
    data = load_feedback()
    data.append({"user": user, "feedback": feedback, "timestamp": datetime.now().isoformat()})
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f)

def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            return json.load(f)
    return []

def get_feedback_stats(feedback_list):
    thumbs_up = sum(1 for f in feedback_list if f["feedback"] == "up")
    thumbs_down = sum(1 for f in feedback_list if f["feedback"] == "down")
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

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="AI Health MVP", layout="wide")
st.title("🌍 AI-Powered Health Assistant for Students")

# ---------------- Disclaimer & Privacy ----------------
st.warning("""
⚠️ **Disclaimer:** The information provided by this AI assistant is for **educational purposes only**. 
It does **not replace professional medical advice**. Always consult a qualified healthcare provider for personal medical concerns.

🔒 **Privacy Notice:** 
- Your data (PDFs, questions, dietary preferences) is processed locally in this app. 
- No personal data is shared externally. 
- Feedback and inputs are stored securely in `feedback.json` only for dashboard purposes.
""")

username = st.text_input("Enter your name:", "Student")

tabs = ["📄 PDF Q&A", "⏳ Time Management", "💊 Health Reminders", "🥗 Nutrition Assistant", "📊 Dashboard"]
choice = st.sidebar.radio("Navigate:", tabs)

# ---------------- Tab 1: PDF Q&A ----------------
if choice == "📄 PDF Q&A":
    st.header("Upload a PDF and Ask Questions")
    st.info("📄 Upload your medical PDF and ask a question. You can download the answer as a PDF.")
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    user_query = st.text_input("Enter your question:")

    if pdf_file and user_query:
        with st.spinner("Extracting and analyzing..."):
            pdf_text = extract_text_from_pdf(pdf_file)
            answer = summarize_with_gemini(pdf_text, user_query)
        st.subheader("Answer")
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
        content = "\n".join([f"{day}: {r}" for day, rlist in DEFAULT_REMINDERS.items() for r in rlist])
        filename = export_to_pdf(content, "tb_schedule.pdf")
        with open(filename, "rb") as f:
            st.download_button("Download PDF", f, file_name="tb_schedule.pdf")

# ---------------- Tab 4: Nutrition Assistant ----------------
elif choice == "🥗 Nutrition Assistant":
    st.header("AI-Powered Nutrition Assistant")
    st.info("🥗 Enter your dietary preferences or health goal and get a personalized meal plan.")
    user_input = st.text_area("Dietary preferences or health goal:")

    if st.button("Get Nutrition Advice"):
        if user_input.strip() == "":
            st.warning("Please enter your preferences or health goal.")
        else:
            with st.spinner("Generating personalized advice..."):
                prompt = f"""
                You are a health assistant specialized in nutrition for TB patients.
                Explain things simply and clearly for a student balancing studies and treatment.
                Generate a daily meal plan and explain why each food choice is beneficial.
                User input: {user_input}
                """
                try:
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    response = model.generate_content(prompt)
                    st.success("Here’s your personalized nutrition advice:")
                    st.write(response.text)

                    if st.button("Download Meal Plan as PDF"):
                        filename = export_to_pdf(response.text, "nutrition_plan.pdf")
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

                except Exception as e:
                    st.error(f"Error generating advice: {e}")

# ---------------- Tab 5: Dashboard ----------------
elif choice == "📊 Dashboard":
    st.header("Your Progress Dashboard")
    st.info("📊 View feedback statistics from AI-generated answers and nutrition advice.")
    feedback_list = load_feedback()
    if feedback_list:
        thumbs_up, thumbs_down = get_feedback_stats(feedback_list)
        st.subheader("👍 AI Feedback Summary")
        st.write(f"Total Feedback Entries: {len(feedback_list)}")
        st.write(f"Positive: {thumbs_up}, Negative: {thumbs_down}")
        st.bar_chart({"👍 Yes": [thumbs_up], "👎 No": [thumbs_down]})
    else:
        st.info("No feedback data yet.")
