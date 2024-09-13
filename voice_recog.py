import os
import time
import speech_recognition as sr
from pydub import AudioSegment
from pydub.generators import Sine
from pydub.playback import play
import subprocess
from tempfile import NamedTemporaryFile
from pydub.utils import get_player_name

def play_suppress_output(audio_segment):
    """
    Play an audio segment without verbose ffmpeg output.
    
    Args:
        audio_segment (AudioSegment): The audio segment to play.
    """
    PLAYER = get_player_name()
    with NamedTemporaryFile("w+b", suffix=".wav") as f:
        audio_segment.export(f.name, "wav")
        with open(os.devnull, 'w') as devnull:
            subprocess.call(
                [PLAYER, "-nodisp", "-autoexit", f.name],
                stdout=devnull,
                stderr=devnull
            )

def play_beep(frequency, duration):
    """
    Play a beep sound.
    
    Args:
        frequency (float): The frequency of the beep in Hz.
        duration (int): The duration of the beep in milliseconds.
    """
    beep = Sine(frequency).to_audio_segment(duration=duration)
    play_suppress_output(beep)

def record_audio(duration=6):
    """
    Record audio from the microphone and transcribe it using Google Speech Recognition.
    
    Args:
        duration (int): The duration of the recording in seconds. Default is 6 seconds.
    
    Returns:
        str: The transcribed text from the audio recording.
    """
    audio_filename = "human_speech.wav"
    
    while True:
        if os.path.exists(audio_filename):
            os.remove(audio_filename)
        
        input("--- Press <ENTER> to start recording ---")
        
        # Start recording
        play_beep(987.77, 100)  # 1000 Hz frequency for 100 ms
        os.system(f"ffmpeg -f alsa -t {duration} -i default -loglevel quiet {audio_filename}")
        play_beep(493.88, 100)  # 500 Hz frequency for 100 ms
        
        # Initialize recognizer
        recognizer = sr.Recognizer()
        
        # Transcribe the recorded audio
        with sr.AudioFile(audio_filename) as source:
            audio = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio)
                print(f"USER INPUT: {text}")
                return text
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
                time.sleep(5)
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
                time.sleep(5)

if __name__ == "__main__":
    # Test the record_audio function
    result = record_audio()
    print(f"Transcribed text: {result}")
