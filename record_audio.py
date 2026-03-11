import os
import sounddevice as sd
import soundfile as sf
import random

# Directory to save recorded audio files
AUDIO_DIR = os.path.join("data", "no_speech") 
SAMPLE_RATE = 48000
# Number of recordings to create
N_RECS = 3000
# Duration range for recordings in seconds                                   
MIN_DURATION = 0.3
MAX_DURATION = 1.0

os.makedirs(AUDIO_DIR, exist_ok=True)

# record audio from microphone
def record_audio(filename, duration, sample_rate=SAMPLE_RATE, device_id=None):
    if device_id is not None:
        sd.default.device = device_id
    print(f"Recording audio for {duration} seconds...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    sf.write(filename, recording, sample_rate)
    print(f"Saved recording to {filename}")

for i in range(N_RECS):
    duration = random.uniform(MIN_DURATION, MAX_DURATION)
    filename = os.path.join(AUDIO_DIR, f"no_speech_{i:04d}.wav")
    record_audio(filename=filename, duration=duration, device_id=2)