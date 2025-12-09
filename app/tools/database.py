import sqlite3
import json
import os

# Database file path
DB_PATH = "student_results.db"

def init_database():
    """Initialize the database with the student_results table"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
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
    
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")

def save_result(student_name, question, answer, grading_json):
    """
    Save a test result to the database.
    
    Args:
        student_name: str - Student's username
        question: str - The question asked
        answer: str - Student's answer
        grading_json: dict - Grading result with grade, scores, advice
    """
    # Ensure database exists
    init_database()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Convert scores dict to JSON string
    scores_str = json.dumps(grading_json.get("scores", {}), ensure_ascii=False)
    
    cursor.execute('''
        INSERT INTO student_results (student_name, question, answer, grade, scores, advice)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        student_name,
        question,
        answer,
        grading_json.get("grade", 0),
        scores_str,
        grading_json.get("advice", "")
    ))
    
    conn.commit()
    conn.close()
    print(f"Result saved for {student_name}")

def get_student_results(student_name):
    """
    Retrieve all test results for a specific student.
    
    Args:
        student_name: str - Student's username
    
    Returns:
        list of tuples: All test results for the student
    """
    init_database()
    
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
    init_database()
    
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

# Initialize database on import
init_database()