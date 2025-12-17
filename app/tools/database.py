import sqlite3
import json
import os

# Database file path
DB_PATH = "student_results.db"

def init_database():
    """Initialize the database with the student_results table"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Table for test results (graded answers)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS student_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_name TEXT,
        question TEXT,
        answer TEXT,
        grade REAL,
        scores TEXT,
        advice TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Table for teaching mode interactions (ungraded Q&A)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS student_teach (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_name TEXT,
        chapter TEXT,
        question TEXT,
        answer TEXT,
        input_tokens INTEGER,
        output_tokens INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")

def save_result(student_name, question, answer, grading_json):
    """
    Save a test result to the database.

    Args:
        student_name: str
        question: str or dict
        answer: str or dict
        grading_json: dict
    """

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ---- Normalize all fields to strings ----
    def normalize(value):
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        elif value is None:
            return ""
        return str(value)

    question_str = normalize(question)
    answer_str = normalize(answer)
    grade = grading_json.get("grade", 0)
    scores_str = json.dumps(grading_json.get("scores", {}), ensure_ascii=False)
    advice_str = normalize(grading_json.get("advice", ""))

    # ---- INSERT ----
    cursor.execute('''
        INSERT INTO student_results (student_name, question, answer, grade, scores, advice)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        student_name,
        question_str,
        answer_str,
        grade,
        scores_str,
        advice_str
    ))

    conn.commit()
    conn.close()
    print(f"Result saved for {student_name}")



def save_teach_interaction(student_name, chapter, question, answer, input_tokens=0, output_tokens=0):
    """
    Save a teaching mode interaction to the database.
    
    Args:
        student_name: str - Student's username
        chapter: str - The chapter the question relates to
        question: str - The student's question
        answer: str - The AI's answer
        input_tokens: int - Number of input tokens (optional)
        output_tokens: int - Number of output tokens (optional)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO student_teach (student_name, chapter, question, answer, input_tokens, output_tokens)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        student_name,
        chapter,
        question,
        answer,
        input_tokens,
        output_tokens
    ))
    
    conn.commit()
    conn.close()
    print(f"Teaching interaction saved for {student_name}")


def get_student_chapter_interactions(student_name, chapter=None):
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Simplified debug - just show which call this is
    if chapter is None:
        print(f"🔍 CALL 1: Fetching ALL chapters for '{student_name}' (for follow-up detection)")
    else:
        print(f"🔍 CALL 2: Fetching chapter '{chapter}' for '{student_name}' (for conversation history)")
    
    if chapter is None:
        cursor.execute('''
            SELECT id, chapter, question, answer, timestamp
            FROM student_teach
            WHERE student_name = ?
            ORDER BY timestamp DESC
        ''', (student_name,))
    else:
        cursor.execute('''
            SELECT id, chapter, question, answer, timestamp
            FROM student_teach
            WHERE student_name = ? AND chapter = ?
            ORDER BY timestamp DESC
        ''', (student_name, chapter))
    
    results = cursor.fetchall()
    print(f"   ✅ Returned {len(results)} results")
    
    conn.close()
    return results

def get_student_results(student_name):
    """
    Retrieve all test results for a specific student.
    
    Args:
        student_name: str - Student's username
    
    Returns:
        list of tuples: All test results for the student
    """
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, question, answer, grade, scores, advice, timestamp
        FROM student_results
        WHERE student_name = ?
        ORDER BY timestamp DESC
    ''', (student_name,))
    
    results = cursor.fetchall()
    conn.close()
    
    return results

def get_all_results():
    """
    Retrieve all test results from the database.
    
    Returns:
        list of tuples: All test results
    """
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, student_name, question, answer, grade, scores, advice, timestamp
        FROM student_results
        ORDER BY timestamp DESC
    ''')
    
    results = cursor.fetchall()
    conn.close()
    
    return results


def get_student_average_grade(student_name):
    conn = sqlite3.connect('student_results.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT AVG(grade) 
        FROM student_results 
        WHERE student_name = ?
    ''', (student_name,))
    
    result = cursor.fetchone()[0]
    conn.close()
    
    return result if result is not None else 0.0

def get_student_chapter_interactions_grouped(student_name):
    """
    Get all teaching interactions grouped by chapter for a student.
    
    Returns:
        dict: {chapter_name: [{"question": "...", "answer": "..."}, ...]}
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT chapter, question, answer, timestamp
        FROM student_teach
        WHERE student_name = ?
        ORDER BY timestamp ASC
    ''', (student_name,))
    
    results = cursor.fetchall()
    conn.close()
    
    chapters = {}
    for row in results:
        chapter, question, answer, timestamp = row
        if chapter not in chapters:
            chapters[chapter] = []
        chapters[chapter].append({
            "question": question,
            "answer": answer,
            "timestamp": timestamp
        })
    
    return chapters


def get_user_history(username, min_score=None):
    """
    Retrieve a user's full test history.
    Optionally filter by minimum grade.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        if min_score is not None:
            cursor.execute("""
                SELECT id, student_name, question, answer, grade, scores, advice
                FROM student_results
                WHERE student_name = ? AND grade <= ?
            """, (username, min_score))
        else:
            cursor.execute("""
                SELECT id, student_name, question, answer, grade, scores, advice
                FROM student_results
                WHERE student_name = ?
            """, (username,))

        print("Executing query for user history")
        rows = cursor.fetchall()
        print(f"Fetched {len(rows)} records for user {username}")
        conn.close()

        history = []
        for row in rows:
            history.append({
                "id": row[0],
                "student_name": row[1],
                "question": row[2],
                "answer": row[3],
                "grade": row[4],
                "advice": row[6],
            })

        return history

    except Exception as e:
        print("Error loading user history:", e)
        return []




# Initialize database on import
init_database()