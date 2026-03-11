import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, Resample
from torch.utils.data import Dataset

class ASRDataset(Dataset):
    def __init__(self, data_df, sample_rate, n_mels):
        self.data_df = data_df
        self.sample_rate = sample_rate

        self.melspec = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=n_mels,
            f_min=20,
            f_max=sample_rate // 2 
        )
        # 
        self.amplitude_to_db = AmplitudeToDB()

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        audio_path = self.data_df.iloc[idx, 0]
        label = self.data_df.iloc[idx, 1]

        # Load audio
        wav, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            wav = Resample(sr, self.sample_rate)(wav)

        # Convert to mel spectrogram
        mel = self.melspec(wav)         # shape: [1, n_mels, time]
        mel = self.amplitude_to_db(mel)

        # Remove channel dimension -> [n_mels, time]
        mel = mel.squeeze(0)

        # Normalize
        mel = (mel - mel.mean()) / (mel.std() + 1e-5)

        # Feature length for packing
        feat_len = mel.shape[-1]

        return mel, label, feat_len