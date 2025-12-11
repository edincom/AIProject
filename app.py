# app.py
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response, stream_with_context
from app.backend.ai import ai_answer_stream, generate_test_question, grade_answer
from app.tools.database import get_user_history
import json
from app.tools.database import get_student_chapter_interactions_grouped

app = Flask(__name__, template_folder="app/templates", static_folder="app/static")

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
        username = data.get("username", "Guest")
        chapter = data.get("chapter", None)  # Get chapter from request

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
                # Pass username and chapter to ai_answer_stream
                for chunk in ai_answer_stream(inputs, username=username, chapter=chapter):
                    token_count += 1
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
    

@app.route('/get_chapter_history', methods=['GET'])
def get_chapter_history():
    """Get all chapters a student has asked questions about"""
    
    username = request.args.get('username', 'Guest')
    chapters = get_student_chapter_interactions_grouped(username)
    
    return jsonify({"chapters": chapters})


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
    
@app.route('/history', methods=['GET'])
def history():
    """
    GET /history?username=Alice&failed_only=true
    Retourne l'historique d'un utilisateur.
    Si failed_only=true alors on filtre les items avec grade < 50.
    """
    try:
        username = request.args.get('username')
        if not username:
            return jsonify({"error": "Missing username parameter"}), 400

        failed_only = request.args.get("failed_only", "false").lower() in ("1", "true", "yes")
        
        # Récupère l'historique complet depuis la DB (implémentation dans database.py)
        # Si get_user_history supporte déjà un paramètre min_score on peut l'utiliser,
        # sinon on filtre côté serveur.
        try:
            # Essayons d'appeler get_user_history avec le paramètre (si implémenté)
            history_data = get_user_history(username, min_score=50 if failed_only else None)
        except TypeError:
            # Si la fonction ne prend qu'un seul argument, on récupère tout et on filtre
            history_data = get_user_history(username)
            if failed_only:
                history_data = [h for h in history_data if (h.get("grade") is not None and int(h.get("grade")) < 50)]
        print(history_data)
        return jsonify({"history": history_data})
    except Exception as e:
        print("Error in /history:", e)
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/retest', methods=['GET'])
def retest():
    """
    GET /retest?question_id=123&username=Alice
    Retourne la question correspondant à question_id (pour recharger exactement la même question).
    """
    try:
        username = request.args.get('username')
        qid = request.args.get('question_id')
        if not username or not qid:
            return jsonify({"error": "Missing username or question_id parameter"}), 400

        # Récupère l'historique complet et cherche l'élément
        try:
            history_data = get_user_history(username)
        except TypeError:
            history_data = get_user_history(username)

        # question_id peut être stocké comme int ou str ; on compare en str pour être safe
        item = next((it for it in history_data if str(it.get("id")) == str(qid)), None)
        if item is None:
            return jsonify({"error": "Question not found"}), 404

        # Retourne la question textuelle (tu peux ajouter d'autres champs si besoin)
        return jsonify({
            "question": item.get("question"),
            "answer": item.get("answer"),
            "grade": item.get("grade"),
            "advice": item.get("advice"),
            "id": item.get("id")
        })
    except Exception as e:
        print("Error in /retest:", e)
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True, threaded=True)