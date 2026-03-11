PyTorch project for Automatic Speech Recognition (ASR) of spoken digits with machine learning. A custom neural network is trained on the AudioMNIST dataset for ASR. The AudioMNIST dataset consists of 30000 short audio recordings of spoken digits (0-9) of 60 different speakers. The project also includes a script for live inference where the audio of a microphone is captured and then passed to a trained model for digit prediction.

# Network Architecture

The architecture of the lightweight neural network is inspired by the DeepSpeech model. Input audio signals are first converted into mel spectrograms, which represent the frequency content of the signal over time. Convolutional layers are used to extract local temporal patterns from the mel spectrograms. Then, bidirectional Gated Recurrent Unit (GRU) layers are used to learn long-term temporal dependencies in the speech signal by processing the audio sequence both forward and backward in time. Finally, a fully connected layer maps the learned feature representation to the output classes, producing the model’s prediction vector.

# Live Inference Demonstration Video

A quick video with sound to demonstrate the live inference of a trained model with a simple to use GUI. On button press the audio of a microphone is recorded for a short period. The audio signal is then passed to the model and its prediction is displayed.

https://github.com/user-attachments/assets/28afed28-cc07-4039-a2f1-2c29b5d4b09d

# Requirements

Requires properly installed FFmpeg for torchaudio.
