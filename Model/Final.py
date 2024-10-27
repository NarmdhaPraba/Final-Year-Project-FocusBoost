import moviepy.editor as mp
import speech_recognition as sr
import librosa
import numpy as np
import soundfile as sf
import pickle
from fuzzywuzzy import fuzz


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
        print("Google Web Speech API could not understand the audio")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results from Google Web Speech API; {e}")
        return ""


def recognize_speech_from_mic():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak something (30 seconds max):")
        audio = recognizer.listen(source, timeout=30)  

    try:
        recognized_text = recognizer.recognize_google(audio)
        return recognized_text.lower()
    except sr.UnknownValueError:
        print("Could not understand audio")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return ""


def compare_texts_fuzzy(actual_text, recognized_text):

    levenshtein_ratio = fuzz.ratio(actual_text, recognized_text)

 
    partial_ratio = fuzz.partial_ratio(actual_text, recognized_text)


    token_set_ratio = fuzz.token_set_ratio(actual_text, recognized_text)

  
    percentage_correct = (levenshtein_ratio + partial_ratio + token_set_ratio) / 3

    return percentage_correct


def main():
 
    video_file = "VID-20240718-WA0006.mp4"


    wav_file = convert_video_to_wav(video_file)

  
    enhanced_wav_file = enhance_and_analyze_audio(wav_file)


    actual_text = recognize_speech_from_file(enhanced_wav_file)
    if actual_text:
        print("Text from video: ", actual_text)
     
        with open('recognized_text.pkl', 'wb') as f:
            pickle.dump(actual_text, f)

     
        recognized_text = recognize_speech_from_mic()

        if recognized_text:
            print(f"Recognized text from mic: {recognized_text}")
            percentage_correct = compare_texts_fuzzy(actual_text, recognized_text)
            print(f"Percentage correctness: {percentage_correct:.2f}%")
        else:
            print("No speech detected or recognized from microphone.")
    else:
        print("No recognizable speech in the video audio.")

if __name__ == "__main__":
    main()
