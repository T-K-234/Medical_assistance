from flask import Flask, render_template, request, redirect, url_for, session, send_file
import os
import keras
from PIL import Image
import numpy as np
import cv2
from PyPDF2 import PdfReader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from fpdf import FPDF
#from Crypto.Cipher import AES
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from db import init_db

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'
init_db()  # Initialize DB

@app.route('/')
def root():
    return redirect('/login')  # Always redirect to login when accessing root

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        try:
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            conn.close()
            return redirect('/login')
        except sqlite3.IntegrityError:
            return "Username already exists!"
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        user = cursor.fetchone()
        conn.close()
        if user and check_password_hash(user[2], password):
            session['user'] = username
            return redirect('/main')
        else:
            return "Invalid credentials!"
    return render_template('login.html')

@app.route('/main')
def main_page():
    if 'user' in session:
        return render_template('main.html', username=session['user'])
    else:
        return redirect('/login')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/login')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/eyeabout')
def eyeabout():
    return render_template('eyeabout.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/dect')
def dect():
    return render_template('dect.html')

@app.route('/dectx')
def dectx():
    return render_template('dectx.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/indexeye')
def indexeye():
    return render_template('indexeye.html')

@app.route('/summarizer')
def summarizer():
    return render_template('sum.html')

# Eye disease classification function
def return_eye_class(img_path):
    try:
        eye_model = keras.models.load_model('model_eye.h5')
        eye_img = Image.open(img_path)
        eye_img_array = np.asarray(eye_img)
        eye_img_resize = cv2.resize(eye_img_array, (224, 224))
        eye_img_reshape = np.reshape(eye_img_resize, (1, 224, 224, 3))
        eye_predictions = eye_model.predict(eye_img_reshape)
        eye_classes = ["Cataract", "diabetic_retinopathy", "Glaucoma", "Normal eye"]
        eye_predicted_class = eye_classes[np.argmax(eye_predictions[0])]
        return eye_predicted_class
    except Exception as e:
        return f"Error in prediction: {e}"

# Result route for eye disease prediction
@app.route('/result_eye', methods=['POST'])
def result_eye():
    try:
        if request.method == 'POST':
            input_image = request.files['input_image']
            save_path = os.path.join('static', input_image.filename)
            input_image.save(save_path)

            output = return_eye_class(save_path)

            info = {
                "Cataract": {
                    "symptoms": [
                        "Blurred or cloudy vision",
                        "Sensitivity to light",
                        "Difficulty seeing at night"
                    ],
                    "precautions": [
                        "Wear sunglasses for UV protection",
                        "Avoid smoking and alcohol",
                        "Eat antioxidant-rich foods"
                    ]
                },
                "Glaucoma": {
                    "symptoms": [
                        "Gradual loss of peripheral vision",
                        "Eye pain or pressure",
                        "Halos around lights"
                    ],
                    "precautions": [
                        "Regular eye check-ups",
                        "Control blood pressure",
                        "Use prescribed eye drops"
                    ]
                },
                "diabetic_retinopathy": {
                    "symptoms": [
                        "Blurred or fluctuating vision",
                        "Spots or floaters in vision",
                        "Dark or empty areas in vision"
                    ],
                    "precautions": [
                        "Control blood sugar levels",
                        "Monitor blood pressure and cholesterol",
                        "Avoid smoking"
                    ]
                },
                "Normal eye": {
                    "symptoms": [],
                    "precautions": []
                }
            }

            disease_info = info.get(output, {"symptoms": [], "precautions": []})

            return render_template(
                "indexeye.html",
                img_path=input_image.filename,
                output=output,
                symptoms=disease_info["symptoms"],
                precautions=disease_info["precautions"]
            )
    except Exception as e:
        return f"An error occurred during the prediction process: {e}"

# Summarization helpers
def summarize_report(report, max_length=500, min_length=200):
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    input_text = "summarize: " + report
    inputs = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

def save_summary_to_pdf(summary, output_filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'I', 12)
    pdf.multi_cell(0, 10, summary)
    pdf.output(output_filename)


# Optional summarization route (commented)
@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        if not os.path.exists('summaries'):
            os.makedirs('summaries')

        if 'pdf' not in request.files:
            return 'No file uploaded', 400

        pdf_file = request.files['pdf']
        if pdf_file.filename == '':
            return 'No selected file', 400

        save_path = os.path.join('uploads', pdf_file.filename)
        pdf_file.save(save_path)

        report_text = extract_text_from_pdf(save_path)

        if report_text:
            summary = summarize_report(report_text, max_length=500, min_length=200)
            output_filename = os.path.join('summaries', f"summary_{pdf_file.filename}")
            save_summary_to_pdf(summary, output_filename)
            return send_file(output_filename, as_attachment=True)
        else:
            return 'Could not extract text from the PDF', 400
    except Exception as e:
        return f"An error occurred: {e}", 500


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
