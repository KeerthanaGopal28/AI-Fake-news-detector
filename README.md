# AI Fake News & Fact Checker

An AI-assisted news credibility assessment tool built with **Python, Streamlit, Gemini 2.5 Flash, and Google Search Grounding**. The application analyzes news claims using current web-based evidence and generates an explainable credibility assessment with supporting sources.

## System Architecture & Execution Flow
# AI Fake News & Fact Checker

AI-assisted news credibility assessment tool built with Python,
Streamlit, Gemini 2.5 Flash, and Google Search Grounding.

## Features
- Real-time web-grounded claim analysis
- Verdict and supporting summary
- Source-based evidence
- 1-hour response caching
- Session-based state management
- Structured response parsing

## System Architecture & Execution Flow

```text
┌──────────────────────────────┐
│       User Input / Claim     │
│      Headline or News Text   │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       Streamlit UI           │
│   Input + Result Dashboard   │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│    Streamlit Cache Layer     │
│       @st.cache_data         │
│          TTL: 1 hour         │
└──────────────┬───────────────┘
               │
        ┌──────┴──────┐
        │             │
    Cache Hit      Cache Miss
        │             │
        ▼             ▼
┌──────────────┐  ┌──────────────────────┐
│ Cached Result│  │   Gemini 2.5 Flash    │
└──────┬───────┘  │     AI Analysis      │
       │          └──────────┬───────────┘
       │                     │
       │                     ▼
       │          ┌──────────────────────┐
       │          │ Google Search        │
       │          │ Grounding            │
       │          │ Current Web Evidence │
       │          └──────────┬───────────┘
       │                     │
       │                     ▼
       │          ┌──────────────────────┐
       │          │ Response Processing  │
       │          │ Verdict • Summary    │
       │          │ Confidence • Sources │
       │          │ Keywords             │
       │          └──────────┬───────────┘
       │                     │
       └──────────┬──────────┘
                  ▼
       ┌──────────────────────┐
       │   Streamlit Dashboard│
       │ Verdict + Evidence   │
       └──────────────────────┘
```


## Evaluation

The system was evaluated using a labeled dataset of **20 news claims**.

| Metric             |     Result |
| ------------------ | ---------: |
| Accuracy           | **89.47%** |
| Weighted Precision | **84.74%** |
| Weighted Recall    | **89.47%** |
| Weighted F1 Score  | **86.98%** |

The evaluation measures how well the system's generated classifications matched the labeled dataset. Results may vary depending on the claims, available web sources, and model responses.

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

## Limitations
- AI-generated assessment is not guaranteed to be factual.
- Search results may vary over time.
- Important claims should be verified using authoritative sources.

## Getting Started

### Prerequisites

* Python **3.10 or higher**
* A **Gemini API key** from Google AI Studio
* Git

### 1. Clone the Repository

```bash
git clone https://github.com/KeerthanaGopal28/AI-Fake-news-detector.git
cd AI-Fake-news-detector
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
AI-Fake-news-detector/
│
├── app.py
├── fact_chcker.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── evaluation/
│   └── evaluate.py
│
└── .streamlit/
    └── secrets.toml
```
## Limitations

- The system provides an AI-assisted assessment and does not guarantee factual accuracy.
- Results depend on the quality and availability of retrieved web sources.
- Model responses may vary between evaluations.
- Important claims should be verified using authoritative sources.

## 📌 Disclaimer

The system provides an **AI-assisted credibility assessment**, not an absolute determination of truth. Users should verify important claims using multiple authoritative sources before relying on the result.
