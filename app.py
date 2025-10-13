import streamlit as st
import google.generativeai as genai
import os
import re
import pandas as pd

# -------------------------------
# CONFIGURE GEMINI API
# -------------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")

# -------------------------------
# Persistent Login Button at Top
# -------------------------------
USERS_FILE = "users.csv"

def save_user(username, password):
    """Save user to CSV if not exists"""
    if os.path.exists(USERS_FILE):
        df = pd.read_csv(USERS_FILE)
        if username not in df['username'].values:
            df = pd.concat([df, pd.DataFrame([{"username": username, "password": password}])], ignore_index=True)
            df.to_csv(USERS_FILE, index=False)
    else:
        pd.DataFrame([{"username": username, "password": password}]).to_csv(USERS_FILE, index=False)

# Login button and form
if 'show_login' not in st.session_state:
    st.session_state['show_login'] = False

if st.button("Login"):
    st.session_state['show_login'] = True

if st.session_state['show_login']:
    st.subheader("Login (Optional)")
    with st.form("login_form", clear_on_submit=False):
        username_input = st.text_input("Username")
        password_input = st.text_input("Password", type="password")
        login_submit = st.form_submit_button("Submit")

    if login_submit:
        if username_input and password_input:
            save_user(username_input, password_input)
            st.session_state['username'] = username_input
            st.success(f"Logged in as: {username_input}")
        else:
            st.warning("Please enter both username and password.")

if 'username' in st.session_state:
    st.info(f"✅ Logged in as: {st.session_state['username']}")

# -------------------------------
# Gemini API / News Analysis
# -------------------------------
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("API key not found! Add it to Streamlit secrets or environment variables.")
else:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("models/gemini-2.5-flash")

    st.subheader("Paste the news article or headline below 👇")
    text = st.text_area("News text:")

    if st.button("Check REAL or FAKE"):
        if not text.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Analyzing with Gemini..."):
                prompt = f"""
You are a news credibility assessor. 
For the following text, output 3 things in this format:
1. Credibility: Real or Fake (add confidence percentage)
2. Short summary (1-2 sentences)
3. Related keywords/topics

Text:
{text}
"""
                try:
                    response = model.generate_content(prompt)
                    output = response.text.strip()

                    # Extract Credibility line for coloring
                    credibility_match = re.search(r"Credibility:\s*(.*)", output, re.IGNORECASE)
                    credibility = credibility_match.group(1) if credibility_match else "Unknown"

                    # Display colored result
                    if "fake" in credibility.lower():
                        st.error(f"⚠️ {credibility}")
                    elif "real" in credibility.lower():
                        st.success(f"✅ {credibility}")
                    else:
                        st.info(f"ℹ️ {credibility}")

                    # Display the full AI output
                    st.markdown("### Full Analysis:")
                    st.write(output)

                    # -------------------------------
                    # Feedback Form (only after analysis)
                    # -------------------------------
                    st.markdown("---")
                    st.subheader("Feedback (Optional)")
                    with st.form("feedback_form"):
                        feedback_text = st.text_area("Was the prediction accurate? Any comments:")
                        feedback_submit = st.form_submit_button("Submit Feedback")

                        if feedback_submit:
                            if feedback_text.strip():
                                # Save feedback to CSV
                                feedback_file = "feedback.csv"
                                user = st.session_state.get('username', 'Anonymous')
                                if os.path.exists(feedback_file):
                                    df_fb = pd.read_csv(feedback_file)
                                    df_fb = pd.concat([df_fb, pd.DataFrame([{"username": user, "feedback": feedback_text}])], ignore_index=True)
                                    df_fb.to_csv(feedback_file, index=False)
                                else:
                                    pd.DataFrame([{"username": user, "feedback": feedback_text}]).to_csv(feedback_file, index=False)

                                st.success("Thank you for your feedback!")
                            else:
                                st.warning("Please enter some feedback before submitting.")

                except Exception as e:
                    st.error(f"Error: {e}")
