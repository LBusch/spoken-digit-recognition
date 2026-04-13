import os
os.add_dll_directory("C:\\ffmpeg\\bin") # Add FFmpeg to PATH for torchaudio
import torch
import torchaudio
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from asr_model import ASRModel
from dataset import ASRDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- Hyperparameters ---
N_MELS = 40
HIDDEN_DIM = 128
GRU_LAYERS = 2
N_LABELS = 11
SAMPLE_RATE = 48000
BATCH_SIZE = 32
LR = 0.0001
NUM_EPOCHS = 5
DATA_CSV = "data/audioMNIST.csv"
EXPERIMENT_NAME = "5_epochs"

# --- Data Preparation ---
data_df = pd.read_csv(DATA_CSV)
labels = data_df['label'].unique().tolist()
label_to_index = {label: idx for idx, label in enumerate(labels)}
train_df, val_df = train_test_split(data_df, test_size=0.2, random_state=42, stratify=data_df['label'])
train_dataset = ASRDataset(data_df=train_df, sample_rate=SAMPLE_RATE, n_mels=N_MELS)
val_dataset = ASRDataset(data_df=val_df, sample_rate=SAMPLE_RATE, n_mels=N_MELS)

# Custom collate function to handle variable-length inputs
def collate_fn(batch):
    mels, labels, feature_lens = zip(*batch)
    mels = [mel.transpose(0, 1) for mel in mels]  # Transpose to (time, mel)
    mels = pad_sequence(mels, batch_first=True)
    labels = [label_to_index[label] for label in labels]
    labels = torch.tensor(labels, dtype=torch.long)
    feature_lens = torch.tensor(feature_lens, dtype=torch.long)
    return mels, labels, feature_lens

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)

# --- Model Initialization ---
model = ASRModel(input_dim=N_MELS, hidden_dim=HIDDEN_DIM, output_dim=N_LABELS, gru_layers=GRU_LAYERS)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = torch.nn.CrossEntropyLoss()

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_losses, val_losses = [], []
train_accs, val_accs = [], []

# --- Training Loop ---
def main():
    print("Training on device:", device)
    for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs", position=0):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", position=1, leave=False)
        for mels, labels, feature_lengths in train_iter:
            # forward pass
            mels, labels, feature_lengths = mels.to(device), labels.to(device), feature_lengths.to(device)
            optimizer.zero_grad()
            outputs = model(mels, feature_lengths)
            # backpropagation
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            # accumulate metrics    
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
            train_iter.set_postfix({"train loss": f"{loss.item():.4f}"})
        
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_correct / train_total)
        
        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", position=1, leave=False)
            for mels, labels, feature_lengths in val_iter:
                mels, labels, feature_lengths = mels.to(device), labels.to(device), feature_lengths.to(device)
                outputs = model(mels, feature_lengths)
                loss = loss_fn(outputs, labels)
                # accumulate metrics      
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
                val_iter.set_postfix({"val loss": f"{loss.item():.4f}"})
        
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(val_correct / val_total)
        
        tqdm.write(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, "
            f"Train Acc: {train_accs[-1]:.4f}, Val Acc: {val_accs[-1]:.4f}\n")
        
    print("Training complete.")

    # --- Save Model and Plot Metrics ---
    experiment_dir = os.path.join("models", EXPERIMENT_NAME)
    os.makedirs(experiment_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(experiment_dir, f"{EXPERIMENT_NAME}_model.pth"))

    # Plot and save figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(train_accs, label="Train Acc")
    ax2.plot(val_accs, label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, f"{EXPERIMENT_NAME}_metrics.png"))
    plt.close()

if __name__ == "__main__":
    main()
