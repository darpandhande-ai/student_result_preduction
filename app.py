import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load model safely with fallback handling
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'svm_model.pkl')
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
else:
    print(f"Warning: Model file not found at {MODEL_PATH}")

# Modern UI HTML/CSS with animations embedded
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predictive Analytics Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            --card-bg: rgba(30, 41, 59, 0.75);
            --primary-accent: #6366f1;
            --secondary-accent: #a855f7;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --border-color: rgba(255, 255, 255, 0.12);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            background-size: 300% 300%;
            animation: gradientShift 12s ease infinite;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
            color: var(--text-main);
        }

        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .container {
            width: 100%;
            max-width: 820px;
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-color);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            animation: slideUp 0.8s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        header {
            text-align: center;
            margin-bottom: 2rem;
        }

        header h1 {
            font-size: 2.2rem;
            background: linear-gradient(90deg, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }

        header p {
            color: var(--text-sub);
            margin-top: 0.5rem;
            font-size: 0.95rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
            gap: 1.25rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
        }

        .input-group label {
            font-size: 0.85rem;
            margin-bottom: 0.4rem;
            color: var(--text-sub);
            font-weight: 600;
        }

        .input-group input, .input-group select {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border-color);
            padding: 0.8rem 1rem;
            border-radius: 12px;
            color: var(--text-main);
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s ease;
        }

        .input-group input:focus, .input-group select:focus {
            border-color: var(--primary-accent);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25);
            transform: translateY(-2px);
        }

        .submit-btn {
            grid-column: 1 / -1;
            margin-top: 1rem;
            background: linear-gradient(90deg, var(--primary-accent), var(--secondary-accent));
            color: white;
            border: none;
            padding: 1rem;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35);
        }

        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(168, 85, 247, 0.5);
        }

        #result {
            margin-top: 2rem;
            padding: 1.5rem;
            border-radius: 16px;
            text-align: center;
            font-size: 1.25rem;
            font-weight: 600;
            display: none;
            animation: bounceIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }

        @keyframes bounceIn {
            0% { transform: scale(0.85); opacity: 0; }
            100% { transform: scale(1); opacity: 1; }
        }

        .result-success {
            background: rgba(34, 197, 94, 0.15);
            border: 1px solid rgba(34, 197, 94, 0.4);
            color: #4ade80;
        }

        .result-error {
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.4);
            color: #f87171;
        }
    </style>
</head>
<body>

    <div class="container">
        <header>
            <h1>Predictive Analytics Dashboard</h1>
            <p>Enter parameters below to classify outcome using the trained SVM Model</p>
        </header>

        <form id="prediction-form" class="form-grid">
            <div class="input-group">
                <label for="gender">Gender</label>
                <select id="gender" name="gender" required>
                    <option value="0">Female</option>
                    <option value="1">Male</option>
                </select>
            </div>

            <div class="input-group">
                <label for="age">Age</label>
                <input type="number" id="age" name="age" value="18" required>
            </div>

            <div class="input-group">
                <label for="study_hours_per_week">Study Hours / Week</label>
                <input type="number" step="0.1" id="study_hours_per_week" name="study_hours_per_week" value="15.0" required>
            </div>

            <div class="input-group">
                <label for="attendance_rate">Attendance Rate (0.0 – 1.0)</label>
                <input type="number" step="0.01" id="attendance_rate" name="attendance_rate" value="0.85" required>
            </div>

            <div class="input-group">
                <label for="parent_education">Parent Education Level</label>
                <select id="parent_education" name="parent_education" required>
                    <option value="0">High School</option>
                    <option value="1">Bachelor</option>
                    <option value="2">Master</option>
                </select>
            </div>

            <div class="input-group">
                <label for="internet_access">Internet Access</label>
                <select id="internet_access" name="internet_access" required>
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>

            <div class="input-group">
                <label for="extracurricular">Extracurricular Activity</label>
                <select id="extracurricular" name="extracurricular" required>
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>

            <div class="input-group">
                <label for="previous_score">Previous Score</label>
                <input type="number" step="0.1" id="previous_score" name="previous_score" value="75.0" required>
            </div>

            <div class="input-group">
                <label for="final_score">Final Exam Score</label>
                <input type="number" step="0.1" id="final_score" name="final_score" value="80.0" required>
            </div>

            <button type="submit" class="submit-btn">Run Prediction</button>
        </form>

        <div id="result"></div>
    </div>

    <script>
        document.getElementById('prediction-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            const resultDiv = document.getElementById('result');
            resultDiv.style.display = 'none';

            const formData = new FormData(this);
            const features = Array.from(formData.values()).map(Number);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ features })
                });

                const result = await response.json();
                
                if (response.ok) {
                    resultDiv.className = 'result-success';
                    resultDiv.innerHTML = `Predicted Outcome: <strong>${result.prediction}</strong>`;
                } else {
                    resultDiv.className = 'result-error';
                    resultDiv.innerHTML = `Error: ${result.error}`;
                }
            } catch (err) {
                resultDiv.className = 'result-error';
                resultDiv.innerHTML = 'Error connecting to the backend server.';
            }

            resultDiv.style.display = 'block';
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model file not found on server.'}), 500

    try:
        data = request.get_json(force=True)
        features = np.array(data['features']).reshape(1, -1)
        
        prediction = model.predict(features)
        
        return jsonify({'prediction': str(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
