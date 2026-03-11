import torch
import torchaudio
import sounddevice as sd
from asr_model import ASRModel
import pandas as pd
import tkinter as tk
from tkinter import ttk
import threading

# --- Configuration ---
SAMPLE_RATE = 48000
DURATION = 1  # seconds
N_MELS = 40
MODEL_PATH = "models/no_speech/no_speech_model.pth"
DATA_CSV = "data/audioMNIST2.csv"
AUDIO_DEVICE_INDEX = None  # Set to None to use default device

# Load labels from CSV
data_df = pd.read_csv(DATA_CSV)
labels = data_df['label'].unique().tolist()

# Load the pre-trained ASR model
model = ASRModel(input_dim=N_MELS, hidden_dim=128, output_dim=11, gru_layers=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
model.to(device)

melspec = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=400,
    win_length=400,
    hop_length=160,
    n_mels=N_MELS,
    f_min=20,
    f_max=SAMPLE_RATE // 2 
) 
amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

# --- GUI Setup ---
root = tk.Tk()
root.title("Live Audio Inference")
window_width = 600
window_height = 300
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
center_x = int(screen_width / 2 - window_width / 2)
center_y = int(screen_height / 2 - window_height / 2)
root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
label = ttk.Label(root, text="Press the button to record audio and perform inference", font=("Helvetica", 14)).pack(pady=20)
style = ttk.Style(root)
style.configure("TButton", font=("Helvetica", 14), padding=10)
prediction_text = tk.StringVar()

def record_audio(duration=DURATION):
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32', device=AUDIO_DEVICE_INDEX)
    sd.wait()
    return audio.flatten()

# Preprocess audio data to model input
def preprocess_audio(audio):
    audio = torch.tensor(audio, dtype=torch.float32, device=device)
    audio = audio.unsqueeze(0)  # Shape: (1, samples)
    mel = melspec(audio)
    mel = amplitude_to_db(mel)
    mel = mel.squeeze(0)  # Remove channel dimension
    mel = (mel - mel.mean()) / (mel.std() + 1e-5)  # Normalize
    feat_len = mel.shape[-1]
    feat_len = torch.tensor(feat_len, dtype=torch.long, device=device)
    feat_len = feat_len.unsqueeze(0)  # Add batch dimension
    mel = mel.transpose(0, 1)  # Shape: (time, n_mels)
    mel = mel.unsqueeze(0)  # Add batch dimension
    return mel, feat_len

# Model inference
def infer(mel, feat_len):
    mel = mel.to(device)
    with torch.no_grad():
        outputs = model(mel, feat_len)
        predicted = torch.argmax(outputs, dim=1)
        confidence = torch.softmax(outputs, dim=1).max()
    return predicted.item(), confidence.item()

# Function to handle recording and inference in a separate thread
def record_and_infer():
    root.after(0, lambda: prediction_text.set("Recording..."))
    audio_data = record_audio(DURATION)
    mel, feat_len = preprocess_audio(audio_data)
    predicted_label, confidence = infer(mel, feat_len)
    # Update GUI text label with prediction
    root.after(0, lambda: prediction_text.set(f"Prediction: {labels[predicted_label]} | Confidence: {confidence:.2f}"))
    # Re-enable the button
    root.after(0, lambda: button.configure(state="normal"))

# Button callback function
def button_callback():
    button.configure(state="disabled")
    # Start a new thread for recording and inference
    threading.Thread(target=record_and_infer, daemon=True).start()

# Button and prediction label
button = ttk.Button(root, text="Start Recording", command=button_callback, style="TButton")
button.pack()
prediction_label = ttk.Label(root, textvariable=prediction_text, font=("Helvetica", 14))
prediction_label.pack(pady=20)


# Main GUI loop
def on_closing():
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()