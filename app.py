from flask import Flask, request, jsonify
import requests, os

app = Flask(__name__)

# Load Gemini API key from environment variables (set in Render dashboard)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

@app.route("/", methods=["GET"])
def home():
    return {"message": "Gemini Backend running on Render 🚀"}

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        query = data.get("query", "")

        # Gemini API endpoint (your model)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        payload = {
            "contents": [{"parts": [{"text": query}]}]
        }
        headers = {"Content-Type": "application/json"}

        # Call Gemini API
        response = requests.post(url, json=payload, headers=headers)
        data = response.json()

        # Extract AI reply
        ai_output = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "No reply")
        )

        return jsonify({"reply": ai_output})

    except Exception as e:
        return jsonify({"error": str(e)})
