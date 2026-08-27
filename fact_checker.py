import os
import json
import re

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
# JSON EXTRACTION
# ============================================================

def extract_json(text: str):

    text = text.strip()

    # Remove markdown code fences if Gemini adds them
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

    # Remove duplicates
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

    config = types.GenerateContentConfig(
        tools=[
            types.Tool(
                google_search=types.GoogleSearch()
            )
        ]
    )

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

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=config
    )

    if not response.text:

        raise ValueError(
            "Gemini returned an empty response."
        )

    result = extract_json(
        response.text
    )

    result["sources"] = (
        extract_grounding_sources(response)
    )

    return result