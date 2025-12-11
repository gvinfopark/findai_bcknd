from flask import Flask, request, session, render_template, jsonify
from flask_cors import CORS
import os
import uuid
from chatbot import MistralChatbot  # Make sure chatbot.py is in the same folder

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"], supports_credentials=True)
app.secret_key = os.urandom(24)

# Fix OpenBLAS if needed
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize chatbot
chatbot = MistralChatbot()

# In-memory stores
session_store = {}
session_user_profiles = {}

@app.route("/", methods=["GET"])
def index():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())

    session_id = session["session_id"]

    # Profile setup
    if session_id not in session_user_profiles:
        session_user_profiles[session_id] = {
            "name": "",
            "age": "",
            "designation": "",
            "preferences": "Audio enabled"
        }

    chat_session = session_store.get(session_id, [])
    return render_template("index.html", history=chat_session)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    session_id = session.get("session_id", str(uuid.uuid4()))
    session["session_id"] = session_id

    # Initialize memory if needed
    chat_session = session_store.setdefault(session_id, [])

    try:
        # Get LangGraph memory
        prev_state = chatbot.compiled_graph.get_state(
            config={"configurable": {"thread_id": session_id}}
        )
        history = prev_state.values.get("history", []) if prev_state else []

        inputs = {
            "question": user_message,
            "context": [],
            "answer": "",
            "history": history,
        }

        # RAG + Graph Execution
        result = chatbot.compiled_graph.invoke(
            inputs, config={"configurable": {"thread_id": session_id}}
        )

        answer = result.get("answer", "").strip()
        
    except Exception as e:
        return jsonify({"error": f"Chatbot error: {str(e)}"}), 500

    # Save chat history
    chat_session.append((user_message, answer, ""))  # audio path placeholder
    chatbot.cleanup_memory()

    return jsonify({
        "question": user_message,
        "answer": answer
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

