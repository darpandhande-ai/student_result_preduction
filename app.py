import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load the trained SVM model
MODEL_PATH = "model.pkl"

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

# HTML + CSS (with modern UI design & smooth animations)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SVM Prediction Hub</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #6C5CE7;
            --primary-glow: rgba(108, 92, 231, 0.4);
            --secondary: #a29bfe;
            --bg-grad-1: #0f0c20;
            --bg-grad-2: #1a103c;
            --card-bg: rgba(255, 255, 255, 0.05);
            --card-border: rgba(255, 255, 255, 0.12);
            --text-light: #ffffff;
            --text-muted: #a0a0c0;
            --accent-green: #00b894;
            --accent-pink: #fd79a8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Poppins', sans-serif;
        }

        body {
            min-height: 100vh;
            background: linear-gradient(135deg, var(--bg-grad-1), var(--bg-grad-2));
            color: var(--text-light);
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem;
            overflow-x: hidden;
            position: relative;
        }

        /* Animated Glowing Background Orbs */
        body::before, body::after {
            content: '';
            position: absolute;
            width: 320px;
            height: 320px;
            border-radius: 50%;
            filter: blur(90px);
            z-index: 0;
            animation: pulseOrb 8s infinite alternate ease-in-out;
        }

        body::before {
            background: var(--primary);
            top: 10%;
            left: 10%;
        }

        body::after {
            background: var(--accent-pink);
            bottom: 10%;
            right: 10%;
            animation-delay: -4s;
        }

        @keyframes pulseOrb {
            0% { transform: scale(1) translate(0, 0); opacity: 0.5; }
            100% { transform: scale(1.3) translate(30px, -30px); opacity: 0.8; }
        }

        /* Glassmorphism Container */
        .container {
            position: relative;
            z-index: 1;
            width: 100%;
            max-width: 850px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
            animation: slideUp 0.8s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(40px); }
            to { opacity: 1; transform: translateY(0); }
        }

        header {
            text-align: center;
            margin-bottom: 2rem;
        }

        header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(90deg, #fff, var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        /* Grid Layout */
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
            gap: 1.25rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .input-group label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: capitalize;
        }

        .input-group input, .input-group select {
            width: 100%;
            padding: 0.75rem 1rem;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 12px;
            color: #fff;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s ease;
        }

        .input-group select option {
            background: #1a103c;
            color: #fff;
        }

        .input-group input:focus, .input-group select:focus {
            border-color: var(--primary);
            background: rgba(255, 255, 255, 0.1);
            box-shadow: 0 0 12px var(--primary-glow);
        }

        .btn-container {
            margin-top: 2rem;
            text-align: center;
        }

        .submit-btn {
            width: 100%;
            padding: 1rem;
            border: none;
            border-radius: 14px;
            background: linear-gradient(90deg, var(--primary), #8c7ae6);
            color: white;
            font-size: 1.05rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 8px 20px var(--primary-glow);
            position: relative;
            overflow: hidden;
        }

        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 28px var(--primary-glow);
        }

        .submit-btn:active {
            transform: translateY(0);
        }

        /* Result Popup Card */
        .result-card {
            margin-top: 2rem;
            padding: 1.5rem;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid var(--card-border);
            text-align: center;
            display: none;
            animation: fadeIn 0.5s ease forwards;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: scale(0.95); }
            to { opacity: 1; transform: scale(1); }
        }

        .result-card h3 {
            font-size: 1.1rem;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }

        .result-card .prediction-text {
            font-size: 2rem;
            font-weight: 700;
            color: var(--accent-green);
            text-shadow: 0 0 15px rgba(0, 184, 148, 0.4);
        }

        /* Loading Spinner */
        .spinner {
            display: none;
            width: 24px;
            height: 24px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 0.8s ease-in-out infinite;
            margin: 0 auto;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>

    <div class="container">
        <header>
            <h1>Predictive Analytics Dashboard</h1>
            <p>Enter parameters below to classify outcome using the trained SVM Model</p>
        </header>

        <form id="predictionForm">
            <div class="form-grid">
                <div class="input-group">
                    <label for="gender">Gender</label>
                    <select id="gender" name="gender" required>
                        <option value="1">Male</option>
                        <option value="0">Female</option>
                    </select>
                </div>
                <div class="input-group">
                    <label for="age">Age</label>
                    <input type="number" id="age" name="age" value="20" required>
                </div>
                <div class="input-group">
                    <label for="study_hours">Study Hours / Week</label>
                    <input type="number" step="0.1" id="study_hours" name="study_hours_per_week" value="15.0" required>
                </div>
                <div class="input-group">
                    <label for="attendance_rate">Attendance Rate (0.0 - 1.0)</label>
                    <input type="number" step="0.01" id="attendance_rate" name="attendance_rate" value="0.85" required>
                </div>
                <div class="input-group">
                    <label for="parent_education">Parent Education Level</label>
                    <select id="parent_education" name="parent_education" required>
                        <option value="0">High School</option>
                        <option value="1">Bachelor's</option>
                        <option value="2">Master's</option>
                        <option value="3">Doctorate</option>
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
            </div>

            <div class="btn-container">
                <button type="submit" class="submit-btn" id="submitBtn">
                    <span id="btnText">Run Prediction</span>
                    <div class="spinner" id="btnSpinner"></div>
                </button>
            </div>
        </form>

        <div class="result-card" id="resultCard">
            <h3>Predicted Outcome</h3>
            <div class="prediction-text" id="predictionResult">-</div>
        </div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();

            const submitBtn = document.getElementById('submitBtn');
            const btnText = document.getElementById('btnText');
            const btnSpinner = document.getElementById('btnSpinner');
            const resultCard = document.getElementById('resultCard');
            const predictionResult = document.getElementById('predictionResult');

            // Show Loading UI
            btnText.style.display = 'none';
            btnSpinner.style.display = 'block';
            resultCard.style.display = 'none';

            const formData = new FormData(this);
            const data = Object.fromEntries(formData.entries());

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const result = await response.json();

                if (result.status === 'success') {
                    predictionResult.innerText = result.prediction;
                } else {
                    predictionResult.innerText = "Error: " + result.message;
                }
            } catch (err) {
                predictionResult.innerText = "Failed to communicate with server.";
            } finally {
                // Restore UI
                btnText.style.display = 'inline';
                btnSpinner.style.display = 'none';
                resultCard.style.display = 'block';
            }
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
        return jsonify({'status': 'error', 'message': 'Model file not found on server.'}), 500

    try:
        data = request.get_json()

        # Features expected order based on model metadata:
        # ['gender', 'age', 'study_hours_per_week', 'attendance_rate', 'parent_education', 
        #  'internet_access', 'extracurricular', 'previous_score', 'final_score']
        features = [
            float(data.get('gender', 0)),
            float(data.get('age', 0)),
            float(data.get('study_hours_per_week', 0)),
            float(data.get('attendance_rate', 0)),
            float(data.get('parent_education', 0)),
            float(data.get('internet_access', 0)),
            float(data.get('extracurricular', 0)),
            float(data.get('previous_score', 0)),
            float(data.get('final_score', 0))
        ]

        input_array = np.array([features])
        prediction = model.predict(input_array)[0]

        return jsonify({
            'status': 'success',
            'prediction': str(prediction)
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
