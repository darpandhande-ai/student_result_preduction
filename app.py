import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load the trained Support Vector Classifier model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# HTML, Animated Styling, and Dynamic JavaScript
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Performance Classifier</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --primary-accent: #6366f1;
            --secondary-accent: #a855f7;
            --text-color: #f8fafc;
            --text-sub: #94a3b8;
            --border-color: rgba(255, 255, 255, 0.1);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Poppins', sans-serif;
        }

        body {
            background: linear-gradient(-45deg, #0f172a, #1e1b4b, #311042, #020617);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
            color: var(--text-color);
        }

        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .container {
            width: 100%;
            max-width: 800px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
            animation: fadeIn 0.8s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
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
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.2rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
        }

        .input-group label {
            font-size: 0.85rem;
            margin-bottom: 0.4rem;
            color: var(--text-sub);
            text-transform: capitalize;
        }

        .input-group input, .input-group select {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border-color);
            padding: 0.75rem 1rem;
            border-radius: 12px;
            color: var(--text-color);
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s ease;
        }

        .input-group input:focus, .input-group select:focus {
            border-color: var(--primary-accent);
            box-shadow: 0 0 12px rgba(99, 102, 241, 0.4);
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
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        }

        .submit-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(168, 85, 247, 0.5);
        }

        #result {
            margin-top: 2rem;
            padding: 1.5rem;
            border-radius: 16px;
            text-align: center;
            font-size: 1.3rem;
            font-weight: 600;
            display: none;
            animation: popIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }

        @keyframes popIn {
            0% { transform: scale(0.8); opacity: 0; }
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
            <h1>SVM Model Prediction</h1>
            <p>Enter the student feature attributes below to generate dynamic classification.</p>
        </header>

        <form id="prediction-form" class="form-grid">
            <div class="input-group">
                <label for="gender">Gender</label>
                <select id="gender" name="gender" required>
                    <option value="0">Female (0)</option>
                    <option value="1">Male (1)</option>
                </select>
            </div>

            <div class="input-group">
                <label for="age">Age</label>
                <input type="number" id="age" name="age" value="18" required>
            </div>

            <div class="input-group">
                <label for="study_hours_per_week">Study Hours/Week</label>
                <input type="number" step="0.1" id="study_hours_per_week" name="study_hours_per_week" value="15" required>
            </div>

            <div class="input-group">
                <label for="attendance_rate">Attendance Rate (%)</label>
                <input type="number" step="0.1" id="attendance_rate" name="attendance_rate" value="85" required>
            </div>

            <div class="input-group">
                <label for="parent_education">Parent Education Level</label>
                <select id="parent_education" name="parent_education" required>
                    <option value="0">High School (0)</option>
                    <option value="1">Bachelor (1)</option>
                    <option value="2">Master (2)</option>
                </select>
            </div>

            <div class="input-group">
                <label for="internet_access">Internet Access</label>
                <select id="internet_access" name="internet_access" required>
                    <option value="1">Yes (1)</option>
                    <option value="0">No (0)</option>
                </select>
            </div>

            <div class="input-group">
                <label for="extracurricular">Extracurricular Activities</label>
                <select id="extracurricular" name="extracurricular" required>
                    <option value="1">Yes (1)</option>
                    <option value="0">No (0)</option>
                </select>
            </div>

            <div class="input-group">
                <label for="previous_score">Previous Score</label>
                <input type="number" step="0.1" id="previous_score" name="previous_score" value="75" required>
            </div>

            <div class="input-group">
                <label for="final_score">Final Score</label>
                <input type="number" step="0.1" id="final_score" name="final_score" value="80" required>
            </div>

            <button type="submit" class="submit-btn">Predict Outcome</button>
        </form>

        <div id="result"></div>
    </div>

    <script>
        document.getElementById('prediction-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            const resultDiv = document.getElementById('result');
            resultDiv.style.display = 'none';

            const formData = new FormData(this);
            const data = {};
            formData.forEach((value, key) => { data[key] = parseFloat(value); });

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ features: Object.values(data) })
                });

                const result = await response.json();
                
                if (response.ok) {
                    resultDiv.className = 'result-success';
                    resultDiv.innerHTML = `Predicted Output: <strong>${result.prediction}</strong>`;
                } else {
                    resultDiv.className = 'result-error';
                    resultDiv.innerHTML = `Error: ${result.error}`;
                }
            } catch (err) {
                resultDiv.className = 'result-error';
                resultDiv.innerHTML = 'Failed to connect to backend server.';
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
    try:
        data = request.get_json(force=True)
        features = np.array(data['features']).reshape(1, -1)
        
        # Run prediction on SVM model
        prediction = model.predict(features)
        
        return jsonify({'prediction': str(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
