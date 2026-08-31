import os
import json
import re
import time
import hashlib

from google import genai
from google.genai import types


# ============================================================
# API CONFIGURATION
# ============================================================

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise RuntimeError(
        "GOOGLE_API_KEY environment variable not found."
    )

client = genai.Client(api_key=api_key)


# ============================================================
# LOCAL EVALUATION CACHE
# ============================================================

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "evaluation",
    "cache"
)

os.makedirs(CACHE_DIR, exist_ok=True)


def get_cache_file(news_text):

    claim_hash = hashlib.sha256(
        news_text.strip().encode("utf-8")
    ).hexdigest()

    return os.path.join(
        CACHE_DIR,
        f"{claim_hash}.json"
    )


def load_cached_result(news_text):

    cache_file = get_cache_file(news_text)

    if not os.path.exists(cache_file):
        return None

    try:

        with open(
            cache_file,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:
        return None


def save_cached_result(news_text, result):

    cache_file = get_cache_file(news_text)

    with open(
        cache_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text: str):

    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    text = text.strip()

    try:

        return json.loads(text)

    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:

        return json.loads(
            text[start:end + 1]
        )

    raise ValueError(
        "Could not find valid JSON in Gemini response."
    )


# ============================================================
# GROUNDING SOURCES
# ============================================================

def extract_grounding_sources(response):

    sources = []

    try:

        if not response.candidates:
            return sources

        candidate = response.candidates[0]

        grounding_metadata = (
            candidate.grounding_metadata
        )

        if not grounding_metadata:
            return sources

        chunks = (
            grounding_metadata.grounding_chunks
            or []
        )

        for chunk in chunks:

            web_source = getattr(
                chunk,
                "web",
                None
            )

            if not web_source:
                continue

            uri = getattr(
                web_source,
                "uri",
                None
            )

            title = getattr(
                web_source,
                "title",
                None
            )

            if uri:

                sources.append({
                    "title": title or "Web Source",
                    "url": uri
                })

    except Exception:
        return sources

    unique_sources = []
    seen = set()

    for source in sources:

        if source["url"] not in seen:

            seen.add(source["url"])
            unique_sources.append(source)

    return unique_sources


# ============================================================
# FACT CHECKING
# ============================================================

def analyze_news_text(news_text: str):

    news_text = news_text.strip()

    if not news_text:
        raise ValueError(
            "News text cannot be empty."
        )

    # --------------------------------------------------------
    # CHECK LOCAL CACHE
    # --------------------------------------------------------

    cached_result = load_cached_result(
        news_text
    )

    if cached_result is not None:

        print("Using cached result.")

        return cached_result


    # --------------------------------------------------------
    # GEMINI CONFIGURATION
    # --------------------------------------------------------

    config = types.GenerateContentConfig(
        tools=[
            types.Tool(
                google_search=types.GoogleSearch()
            )
        ]
    )


    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are an expert AI-assisted fact-checking system.

Verify the following news claim using CURRENT information
retrieved through Google Search.

Break the statement into individual factual claims.

Verify each claim independently.

Pay special attention to:

- Numbers
- Dates
- Names
- Locations
- Statistics
- Quotes
- Recent events

Do not invent evidence.

If evidence is insufficient, use UNVERIFIED.

Return ONLY valid JSON.

Do not use markdown.
Do not use ```json.

Use exactly this structure:

{{
    "verdict": "UNVERIFIED",
    "confidence": 50,
    "summary": "Short explanation.",
    "claim_analysis": [
        {{
            "claim": "Individual claim",
            "status": "UNVERIFIED",
            "explanation": "Evidence-based explanation."
        }}
    ],
    "evidence": [
        "Important evidence."
    ],
    "keywords": [
        "keyword1",
        "keyword2",
        "keyword3"
    ]
}}

Allowed verdicts:

REAL
MOSTLY REAL
PARTIALLY TRUE
MISLEADING
MOSTLY FALSE
FAKE
UNVERIFIED

Allowed claim statuses:

SUPPORTED
CONTRADICTED
PARTIALLY_SUPPORTED
UNVERIFIED

Confidence must be between 0 and 100.

NEWS CLAIM:

{news_text}
"""


    # --------------------------------------------------------
    # API CALL WITH RETRY
    # --------------------------------------------------------

    max_retries = 4

    for attempt in range(max_retries):

        try:

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=config
            )

            break

        except Exception as e:

            error_message = str(e)

            if "429" not in error_message and "503" not in error_message:
                raise

            if attempt == max_retries - 1:

                raise

            wait_time = 20 * (2 ** attempt)

            print(
                f"Rate limit reached. "
                f"Waiting {wait_time} seconds..."
            )

            time.sleep(wait_time)


    # --------------------------------------------------------
    # PARSE RESPONSE
    # --------------------------------------------------------

    if not response.text:

        raise ValueError(
            "Gemini returned an empty response."
        )

    result = extract_json(
        response.text
    )


    # --------------------------------------------------------
    # ADD GROUNDING SOURCES
    # --------------------------------------------------------

    result["sources"] = (
        extract_grounding_sources(response)
    )


    # --------------------------------------------------------
    # SAVE RESULT TO CACHE
    # --------------------------------------------------------

    save_cached_result(
        news_text,
        result
    )

    print("Result cached.")

    return result