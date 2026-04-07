PyTorch project for Automatic Speech Recognition (ASR) of spoken digits with machine learning. A custom neural network is trained on the AudioMNIST dataset [^1] for ASR. The AudioMNIST dataset consists of 30000 short audio recordings of spoken digits (0-9) of 60 different speakers. The model achieves an accuracy of 99.2% on the dataset. The project also includes a script for live inference where the audio of a microphone is captured and then passed to a trained model for digit prediction.

# Network Architecture

The architecture of the lightweight neural network is inspired by the DeepSpeech [^2] model. Input audio signals are first converted into mel spectrograms, which represent the frequency content of the signal over time. Convolutional layers are used to extract local temporal patterns from the mel spectrograms. Then, bidirectional Gated Recurrent Unit (GRU) layers are used to learn long-term temporal dependencies in the speech signal by processing the audio sequence both forward and backward in time. Finally, a fully connected layer maps the learned feature representation to the output classes, producing the model’s prediction vector.

# Live Inference Demonstration Video

A quick video with sound to demonstrate the live inference of a trained model with a simple to use GUI. On button press the audio of a microphone is recorded for a short period. The audio signal is then passed to the model and its prediction is displayed.

https://github.com/user-attachments/assets/28afed28-cc07-4039-a2f1-2c29b5d4b09d

# How to Use

`organise_data.py`: In case of downloading the AudioMNISt dataset from kaggle, this script reorganizes the file structure into subdirectories for each digit.

`record_audio.py`: Script for recording your own audio samples that can be used for training.

`create_csv.py`: Creates a CSV file containing the file paths and labels of the dataset's audio samples that is used for training.

`train_model.py`: Script for training the model defined in `asr_model.py` on the AudioMNIST dataset.

`live_inference.py`:  Script for live inference of a trained model for ASR of spoken digits with a simple GUI using the audio recording of a microphone. 

# Requirements

Requires properly installed FFmpeg for torchaudio.

## References

[^1]: Becker, S., Vielhaben, J., Ackermann, M., Müller, K., Lapuschkin, S., & Samek, W. (2024) *AudioMNIST: Exploring Explainable Artificial Intelligence for audio analysis on a simple benchmark* Journal of the Franklin Institute, 361(1), 418–428.
DOI: [10.1016/j.jfranklin.2023.11.038](https://doi.org/10.1016/j.jfranklin.2023.11.038)

[^2]: Hannun, A., Case, C., Casper, J., Catanzaro, B., Diamos, G., Elsen, E., Prenger, R., Satheesh, S., Sengupta, S., Coates, A., & Ng, A. Y. (2014). *Deep Speech: Scaling up end-to-end speech recognition* arXiv.org. https://arxiv.org/abs/1412.5567  
DOI: [10.48550/arXiv.1412.5567](https://doi.org/10.48550/arXiv.1412.5567)
