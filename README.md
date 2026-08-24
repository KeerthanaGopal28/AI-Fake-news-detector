# Real-Time AI Fake News & Fact Checker

An enterprise-ready news credibility assessor built with **Python, Streamlit, and Google's Gemini 2.5 Flash model**. The application leverages **live Google Search Grounding** to evaluate real-time claims against current web information, helping overcome static LLM knowledge cutoffs and reduce AI hallucinations.

## System Architecture & Execution Flow

```text
[ User Input / Headline ]
          │
          ▼
[ Streamlit UI Dashboard ]
          │
          ▼
[ Streamlit Caching Layer (@st.cache_data) ]
          │
          ├── (Cache Hit) ──► [ Fast Cached UI Render ]
          │
          └── (Cache Miss)
                    │
                    ▼
          [ Gemini 2.5 Flash Engine ]
                    │
                    ▼
          [ Google Search Grounding ]
                    │
                    ▼
          [ Structured Response Parsing ]
                    │
                    ▼
          [ UI State Rendering ]
          (Verdict, Summary, Keywords)
```

## Key Features

* **Real-Time Web Grounding:** Uses Google's native search retrieval capabilities to evaluate claims against current web information.
* **Intelligent Query Caching:** Uses Streamlit's `@st.cache_data` with a **1-hour TTL** to reduce redundant API requests and improve response time.
* **In-Memory Session Management:** Uses `st.session_state` for session persistence without storing sensitive information in flat files.
* **Dynamic Visual Statusing:** Uses regex pattern matching to extract confidence values and display formatted status alerts using `st.success()`, `st.error()`, and `st.info()`.
* **Structured AI Responses:** Parses the Gemini response into meaningful verdict, summary, confidence, and keyword information.

## Tech Stack

| Component            | Technology                   |
| -------------------- | ---------------------------- |
| Frontend Framework   | Streamlit                    |
| AI Engine            | Google Gemini 2.5 Flash      |
| AI SDK               | `google-genai`               |
| Fact Retrieval       | Google Search Grounding      |
| Programming Language | Python 3.10+                 |
| Caching              | Streamlit `@st.cache_data`   |
| Session Management   | Streamlit `st.session_state` |

## Getting Started

### Prerequisites

* Python **3.10 or higher**
* A **Gemini API key** from Google AI Studio
* Git

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detector.git
cd fake-news-detector
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

Create a `.streamlit/secrets.toml` file in the project root:

```toml
GOOGLE_API_KEY = "your_actual_gemini_api_key_here"
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will start locally and can be accessed through the Streamlit URL displayed in the terminal.

## Security & Environment Notes

* The `.streamlit/secrets.toml` file should be included in `.gitignore` to prevent API keys from being committed to public repositories.
* API credentials are stored through Streamlit secrets rather than hard-coded in the application.
* User inputs and session-specific information are maintained using Streamlit's temporary `st.session_state`.
* **Never expose or commit your Gemini API key to GitHub or other public repositories.**

## Project Structure

```text
fake-news-detector/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── .streamlit/
    └── secrets.toml
```

## 📌 Disclaimer

The system provides an **AI-assisted credibility assessment**, not an absolute determination of truth. Users should verify important claims using multiple authoritative sources before relying on the result.
