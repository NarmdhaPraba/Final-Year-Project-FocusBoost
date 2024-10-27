from flask import Flask, render_template, request, redirect, url_for, send_file , session , flash
import moviepy.editor as mp
import speech_recognition as sr
import librosa
import numpy as np
import soundfile as sf
import pickle
from fuzzywuzzy import fuzz
import os
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.automap import automap_base
from datetime import datetime



app = Flask(__name__)
app.secret_key = 'Focus3510#45'


app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root@localhost/boostfocus'



app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

Base = automap_base()


with app.app_context():
    Base.prepare(db.engine, reflect=True)


Users = Base.classes.users




# Function to convert video to WAV audio
app.config['UPLOAD_FOLDER'] = 'static/uploads'

def convert_video_to_wav(video_file):
    video = mp.VideoFileClip(video_file)
    audio = video.audio
    wav_file = "converted_audio.wav"
    audio.write_audiofile(wav_file)
    return wav_file

def enhance_and_analyze_audio(wav_file, output_file="enhanced_audio.wav"):
    y, sr = librosa.load(wav_file, sr=None)
    D = np.abs(librosa.stft(y))
    spectrogram = librosa.amplitude_to_db(D, ref=np.max)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    y = librosa.util.normalize(y)
    sf.write(output_file, y, sr)
    with open('audio_features.pkl', 'wb') as f:
        pickle.dump({'spectrogram': spectrogram, 'mfccs': mfccs}, f)
    return output_file

def recognize_speech_from_file(wav_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_file) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Could not understand the audio"
    except sr.RequestError as e:
        return f"Request error: {e}"

def recognize_speech_from_mic():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        audio = recognizer.listen(source, timeout=30)
    try:
        return recognizer.recognize_google(audio).lower()
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Request error: {e}"

def compare_texts_fuzzy(actual_text, recognized_text):
    levenshtein_ratio = fuzz.ratio(actual_text, recognized_text)
    partial_ratio = fuzz.partial_ratio(actual_text, recognized_text)
    token_set_ratio = fuzz.token_set_ratio(actual_text, recognized_text)
    percentage_correct = (levenshtein_ratio + partial_ratio + token_set_ratio) / 3
    return percentage_correct

@app.route('/main_upload')
def main_upload():
    return render_template('Upload_video.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'video_file' not in request.files:
        return redirect(request.url)
    file = request.files['video_file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        wav_file = convert_video_to_wav(file_path)
        enhanced_wav_file = enhance_and_analyze_audio(wav_file)
        actual_text = recognize_speech_from_file(enhanced_wav_file)
        return render_template('results.html', actual_text=actual_text, video_file=file.filename)
    return redirect(request.url)

@app.route('/compare', methods=['POST'])
def compare():
    actual_text = request.form['actual_text']
    recognized_text = recognize_speech_from_mic()
    percentage_correct = compare_texts_fuzzy(actual_text, recognized_text)
    return render_template('results.html', actual_text=actual_text, recognized_text=recognized_text, percentage_correct=percentage_correct  )






@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        try:
            Users_ref = db.session.query(Users).filter_by(email=username).first()

            if Users_ref is not None:
                if Users_ref.password == password:
                    session['UserId'] = Users_ref.id
                    session['username'] = username
                    flash('Login successful!', 'success')
                    return redirect(url_for('main_upload'))

                else:
                    flash('Login failed: Invalid password', 'error')
            else:
                flash('Login failed: User not found', 'error')

        except Exception as e:
            flash(f'Login failed: {e}', 'error')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        full_name = request.form.get('Full_Name')
        contact = request.form.get('Contact')
        age = request.form.get('Age')
        email = request.form.get('Email')
        password = request.form.get('Password')

        user_data = {
            'full_name': full_name,
            'contact': contact,
            'age': age,
            'email': email,
            'password': password
        }

        try:
            new_user = Users(**user_data)  # Ensure your Users model has these fields
            db.session.add(new_user)
            db.session.commit()
            flash('Thank you! User registered successfully!', 'success')
        #    return redirect(url_for('visitor_registration'))
        except Exception as e:
            flash(f'Error saving  info: {e}', 'error')

    return render_template('Register.html')


@app.route('/logout')
def logout():
    session.pop('UserId', None)
    session.pop('username', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
