# app.py
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response, stream_with_context
from app.backend.ai import ai_answer_stream, generate_test_question, grade_answer
import json

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
        mode = data.get("mode", "teach")

        # Ensure message is a string
        if not isinstance(message, str):
            message = str(message)
        message = message.strip()

        if not message:
            return jsonify({"reply": "Please enter a message."})

        # Pass the message as a dictionary with 'question' key
        inputs = {"question": message}
        
        # Return a streaming response
        def generate():
            try:
                token_count = 0
                for chunk in ai_answer_stream(inputs):
                    token_count += 1
                    # Debug: print what we're sending
                    # print(f"Sending token #{token_count}: {repr(chunk)[:100]}")
                    # Send each chunk as a JSON object
                    yield f"data: {json.dumps({'token': chunk})}\n\n"
                print(f"Streaming complete. Total tokens: {token_count}")
            except Exception as e:
                print(f"Streaming error: {e}")
                import traceback
                traceback.print_exc()
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )

    except Exception as e:
        print("An error occurred in /chat_api:", e)
        import traceback
        traceback.print_exc()
        return jsonify({"reply": "An error occurred with AI backend."}), 500


@app.route('/test_api', methods=['POST'])
def test_api():
    """Handle test mode - generate question and grade answer"""
    try:
        data = request.get_json() or {}
        action = data.get("action", "")  # "generate" or "grade"
        
        if action == "generate":
            # Generate a test question based on criteria
            criteria = data.get("criteria", "")
            
            if not criteria:
                return jsonify({"error": "Please specify test criteria"}), 400
            
            # Generate question using RAG
            question = generate_test_question(criteria)
            
            return jsonify({
                "question": question,
                "success": True
            })
        
        elif action == "grade":
            # Grade the student's answer
            question = data.get("question", "")
            answer = data.get("answer", "")
            rubric = data.get("rubric", "")
            username = data.get("username", "Anonymous")
            
            if not all([question, answer, rubric]):
                return jsonify({"error": "Missing required fields"}), 400
            
            # Grade the answer
            grading_result = grade_answer(question, answer, rubric, username)
            
            return jsonify({
                "grading": grading_result,
                "success": True
            })
        
        else:
            return jsonify({"error": "Invalid action"}), 400
    
    except Exception as e:
        print("An error occurred in /test_api:", e)
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True, threaded=True)