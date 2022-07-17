import torch
import torch.nn.functional as F
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import pickle
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np

# load in file location
parser = argparse.ArgumentParser(description='Test RT60',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('test_path', help='Destination of test files')
args = parser.parse_args()

data_direc = Path(args.test_path)
# send to cuda line
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load in classifier classes, saved model and put into eval mode
classifier_classes = pickle.load(open('results/classes', 'rb'))
model = torch.load('2s_rt60_classifier.pt')
model.eval()
if torch.cuda.is_available():
        model.cuda()
# set up values for spec as hyperparams
sample_rate = 48000  # sample rate needs to match audio files
n_fft = 800  # size of FFT, default 400
normalized = False  # normalize after FFT, can be True or False
transform = T.Spectrogram(n_fft=n_fft, normalized=normalized)

# set up function to generate spectrograms
def process_spectrogram_from_file(audio_path, sample_rate):
    waveform, sr = torchaudio.load(audio_path)
    waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    spec = transform(waveform)
    return spec

# iterate through files in directory
results = {}

for path in sorted(data_direc.glob('*.wav')):
    print(path)
    spec = process_spectrogram_from_file(path, 48000).to(device)
    prob = F.softmax(model(spec.unsqueeze(0)), dim=1)
    pred_class = classifier_classes[int(prob.argmax())]
    results[str(path.stem)] = pred_class

# print amount of each class detected
from pprint import pprint
pprint(results)

