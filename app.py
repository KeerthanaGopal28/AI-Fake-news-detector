import os
import re
import streamlit as st
from google import genai
from google.genai import types

# 1. Page Configuration
st.set_page_config(
    page_title="AI Fake News & Fact Checker", 
    page_icon="📰", 
    layout="wide"
)

st.title("📰 Real-time AI Fake News & Fact Checker")
st.caption(
    "Grounded with Google Search to analyze news authenticity and context in real-time."
)

# 2. Gemini API Setup (Google GenAI SDK)
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error(
        "⚠️ API key not found! Please configure `GOOGLE_API_KEY` in Streamlit secrets or environment variables."
    )
else:
    # Initialize Client using the modern SDK
    client = genai.Client(api_key=api_key)

    @st.cache_data(ttl=3600, show_spinner=False)
    def analyze_news_text(news_text: str):
        # Configure Live Search Grounding
        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]
        )

        prompt = f"""
You are an expert news credibility assessor. 
Analyze the provided news text using live search grounding to verify recent and real-time facts up to the present day.

Provide your response strictly in the following structured format:
1. CREDIBILITY: [REAL / FAKE / UNVERIFIED] (Confidence: [X]%)
2. SUMMARY: [1-2 sentences summarizing why it is real or fake based on live search evidence]
3. KEYWORDS: [3-5 key topics/entities separated by commas]

Text to analyze:
{news_text}
"""
        # Call Gemini 2.5 Flash model with search grounding config
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt, 
            config=config
        )
        return response.text.strip()

    # 3. Main Interface & Input
    st.subheader("Paste the News Article or Headline Below 👇")
    text_input = st.text_area("News Text:", height=180, placeholder="Paste a headline or short paragraph to analyze...")

    if st.button("🔍 Check Credibility", type="primary"):
        if not text_input.strip():
            st.warning("Please enter news text to analyze.")
        else:
            with st.spinner("Cross-referencing live search results and verifying sources..."):
                try:
                    output = analyze_news_text(text_input)

                    # Extract Credibility verdict via regex
                    credibility_match = re.search(
                        r"CREDIBILITY:\s*(.*)", output, re.IGNORECASE
                    )
                    credibility = (
                        credibility_match.group(1)
                        if credibility_match
                        else "UNVERIFIED"
                    )

                    # Dynamic Visual Indicators
                    if "fake" in credibility.lower():
                        st.error(f"🚨 **Verdict:** {credibility}")
                    elif "real" in credibility.lower():
                        st.success(f"✅ **Verdict:** {credibility}")
                    else:
                        st.info(f"ℹ️ **Verdict:** {credibility}")

                    # Detailed Analysis Display
                    st.markdown("### 📋 Analysis Breakdown")
                    st.write(output)

                except Exception as e:
                    st.error(f"Analysis failed: {e}")