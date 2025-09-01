import speech_recognition as sr
from gtts import gTTS
import os
import io
import tempfile
from pydub import AudioSegment
from pydub.playback import play as pydub_play

try:
    import whisper
except ImportError:
    whisper = None
    print("Warning: 'openai-whisper' not installed. Local transcription is unavailable.")


# --- Text to Speech (Robust Version) ---
def text_to_speech(text):
    """Converts text to speech and plays it in memory using pydub."""
    print(f"--- 🔊 Converting text to speech... ---")
    try:
        mp3_fp = io.BytesIO()
        tts = gTTS(text=text, lang='en')
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)

        audio = AudioSegment.from_file(mp3_fp, format="mp3")
        print("--- Playing response... ---")
        pydub_play(audio)
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
        print("Agent Response:", text)


# --- Speech to Text (Local Whisper Model) ---
whisper_model = None
if whisper:
    print("--- 🧠 Loading local Whisper model ('base'). This may take a moment... ---")
    whisper_model = whisper.load_model("base")
    print("--- ✅ Whisper model loaded. ---")


def transcribe_audio():
    """Captures and transcribes audio using a local Whisper model."""
    if not whisper_model:
        raise ImportError("Local Whisper model is not available.")

    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 2.0

    with sr.Microphone() as source:
        print("\nSay something!")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio_data = recognizer.listen(source)
        print("Recognizing...")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            temp_audio_file.write(audio_data.get_wav_data())
            filepath = temp_audio_file.name
        result = whisper_model.transcribe(filepath, fp16=False)
        os.remove(filepath)
        return result["text"].strip()
    except Exception as e:
        print(f"Error with local Whisper model: {e}")
        return ""
