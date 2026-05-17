from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import joblib
import sqlite3

app = Flask(__name__)
app.secret_key = "heart_secret_key"
def init_db():
    
    conn = sqlite3.connect('users.db')

    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    ''')

    conn.commit()
    conn.close()

init_db()
# Load model and scaler
model = joblib.load(r"C:\Users\sanch\Downloads\hdp\model.pkl")
scaler = joblib.load(r"C:\Users\sanch\Downloads\hdp\scaler.pkl")


# ---------------- LOGIN PAGE ---------------- #

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def do_login():

    username = request.form['username']
    password = request.form['password']

    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, password)
    )

    user = cursor.fetchone()

    conn.close()

    if user:
        session['user'] = username
        return redirect(url_for('home'))

    return "Invalid Username or Password"

#----------- register----------- #

@app.route('/register')
def register():
    return render_template('register.html')


@app.route('/register_user', methods=['POST'])
def register_user():

    username = request.form['username']
    password = request.form['password']

    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    try:

        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, password)
        )

        conn.commit()

        return redirect(url_for('login'))

    except:
        return "Username already exists"

    finally:
        conn.close()

# ---------------- HOME PAGE ---------------- #

@app.route('/home')
def home():

    if 'user' not in session:
        return redirect(url_for('login'))

    return render_template('index.html')

# ---------------- PREDICTION ---------------- #

@app.route('/predict', methods=['POST'])
def predict():

    if 'user' not in session:
        return redirect(url_for('login'))
    
    try:
        data = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalach']),
            float(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['ca']),
            float(request.form['thal'])
        ]

        input_data = np.array([data])

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)

        if prediction[0] == 1:
            result = "⚠️ High Risk of Heart Disease"
        else:
            result = "✅ Low Risk of Heart Disease"

        return render_template(
            'index.html',
            prediction_text=result
        )

    except Exception as e:
        return str(e)

    


# ---------------- LOGOUT ---------------- #

@app.route('/logout')
def logout():

    session.pop('user', None)

    return redirect(url_for('login'))
if __name__ == "__main__":
    app.run(debug=True)

    #http://127.0.0.1:5000/