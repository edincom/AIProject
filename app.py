# app.py
from flask import Flask, render_template, request, redirect, url_for, jsonify
from app.backend.ai import ai_answer

app = Flask(__name__, template_folder="app/templates")

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        return redirect(url_for('chat', username=username))
    return render_template('login.html')

@app.route('/chat')
def chat():
    username = request.args.get('username', 'Guest')
    return render_template('chat.html', username=username)

@app.route('/chat_api', methods=['POST'])
def chat_api():
    try:
        data = request.get_json() or {}
        message = data.get("message", "")
        mode = data.get("mode", "teach")  # default to teach

        # Ensure message is a string
        if not isinstance(message, str):
            message = str(message)
        message = message.strip()

        if not message:
            return jsonify({"reply": "Please enter a message."})

        # Prepare input for AI backend
        inputs = {"question": message, "mode": mode}

        # Call AI backend
        response = ai_answer(inputs)  # <-- make sure ai_answer handles "mode"
        
        # Convert to string if response is AIMessage or other object
        if hasattr(response, "content"):
            response_text = str(response.content)
        else:
            response_text = str(response)

        return jsonify({"reply": response_text})

    except Exception as e:
        print("An error occurred in /chat_api:", e)
        return jsonify({"reply": "An error occurred with AI backend."}), 500

if __name__ == '__main__':
    app.run(debug=True)
