from flask import Flask, request, jsonify
import os, json, re, requests

app = Flask(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # set in Render

# ---- simple grounding store -------------------------------------------------
def load_courses():
    path = os.path.join(os.path.dirname(__file__), "courses.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

COURSES = load_courses()

def find_relevant_courses(q: str):
    q_lower = q.lower()
    hits = []
    for c in COURSES:
        name = c.get("name", "")
        tags = " ".join(c.get("tags", []))
        if name and (name.lower() in q_lower or any(t.lower() in q_lower for t in c.get("tags", []))):
            hits.append(c)
    return hits[:3]  # keep prompt compact

def courses_to_facts(courses):
    if not courses:
        return "No matching course facts found."
    facts = []
    for c in courses:
        parts = [
            f"Name: {c.get('name','')}",
            f"Syllabus: {c.get('syllabus','N/A')}",
            f"Duration: {c.get('duration','N/A')}",
            f"Fees: {c.get('fees','N/A')}",
            f"Placement: {c.get('placement','N/A')}",
            f"Mode: {c.get('mode','N/A')}",
        ]
        facts.append(" • " + " | ".join(parts))
    return "\n".join(facts)

# ---- routes -----------------------------------------------------------------
@app.route("/", methods=["GET"])
def health():
    return {"message": "Gemini Backend running on Render 🚀"}

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(force=True, silent=True) or {}
        query = (data.get("query") or data.get("message") or data.get("text") or "").strip()
        lang = (data.get("lang") or "en").lower()  # "en" or "hi"

        if not query:
            return jsonify({"error": "No query provided"}), 400

        # Build grounded context
        matched = find_relevant_courses(query)
        facts = courses_to_facts(matched)

        # System guidance to reduce hallucinations
        system_prompt = (
            "You are an EdTech Course Consultant voice agent for an Indian audience.\n"
            f"Respond in {'English' if lang=='en' else 'Hindi'} with a clear, friendly tone.\n"
            "STRICT RULES:\n"
            "• provide accurate information when discussing syllabus, fees, duration, placement.\n"
            "• If a fact is missing or uncertain, say you don’t know and offer to arrange a callback.\n"
            "• Be concise; use bullet points when listing items.\n"
            "• If the query is vague, ask exactly one clarifying question.\n"
            "• Never invent numbers, claims, colleges or guarantees."
        )

        # Gemini 2.0 Flash API
        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        )
        payload = {
            "system_instruction": {
                "role": "system",
                "parts": [{"text": system_prompt + "\n\nFACTS:\n" + facts}]
            },
            "contents": [
                {"role": "user", "parts": [{"text": query}]}
            ],
            "generationConfig": {
                "temperature": 0.2,     # less creative → fewer hallucinations
                "topP": 0.8,
                "topK": 40,
                "maxOutputTokens": 512
            }
        }
        headers = {"Content-Type": "application/json"}

        r = requests.post(url, json=payload, headers=headers, timeout=30)
        data = r.json()

        # Safe extraction
        reply = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text")
        )

        if not reply:
            # fall back to a safe message if API shape changes or empty
            reply = ("I couldn’t find reliable details for that. "
                     "Would you like me to schedule a callback with a human consultant?")

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
