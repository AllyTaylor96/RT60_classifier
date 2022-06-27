"""
This program aims to take a folder of audio files (say the Alba corpus), and
generate spectrograms for the files. It tries to cut down on the storage
requirements by generating the spectrograms for the convolved audio files
directly instead of saving them all as audio files: doing the convolution and
spectro generation in a single step.
"""

import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from utils import *
from pathlib import Path
import argparse

# set up parser
parser = argparse.ArgumentParser(description='Spectrogram generator',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('audio_src', help='Destination of audio files')
parser.add_argument('ir_src', help='Destination of IRs')
parser.add_argument('output_path', help='Destination of IRs')
args = parser.parse_args()

# get paths for where audio data is currently
audio_filepath = Path(args.audio_src)
audio_filenames = sorted(audio_filepath.glob('alba*.wav'))

# split into test and dev sets
train_size = int(0.9 * len(audio_filenames))
test_size = len(audio_filenames) - train_size
train_files = audio_filenames[0:train_size]
test_files = audio_filenames[train_size:]

# set paths for where irs and stored data should go
ir_filepath = Path(args.ir_src)
ir_filenames = sorted(ir_filepath.glob('*.wav'))
ir_filenames = [x for x in ir_filenames if not x.name.startswith('.')]
spec_path = Path(args.output_path)

# set up values for the spectrogram as hyperparams
sample_rate = 24000  # sample rate needs to match audio files
n_fft = 800  # size of FFT, default 400
normalized = False  # normalize after FFT, can be True or False
transform = T.Spectrogram(n_fft=n_fft, normalized=normalized)

def process_ir(ir_filepath, desired_sr):
    waveform, sr = torchaudio.load(ir_filepath)
    waveform = torchaudio.functional.resample(waveform, sr, desired_sr)
    mono_ir_waveform = torch.mean(waveform, dim=0).unsqueeze(0)
    norm_ir_waveform = mono_ir_waveform / torch.norm(mono_ir_waveform, p=2)
    prepped_ir_waveform = torch.flip(norm_ir_waveform, [1])
    return prepped_ir_waveform


def process_speech(speech_filepath, desired_sr, ir_waveform):
    waveform, sr = torchaudio.load(speech_filepath)
    waveform = torchaudio.functional.resample(waveform, sr, desired_sr)
    padded_src_waveform = torch.nn.functional.pad(waveform,(ir_waveform.shape[1] - 1, 0))
    padded_src_waveform = torch.nn.functional.pad(padded_src_waveform, (0, ir_waveform.shape[1] - 1))
    convolved_speech_waveform = torch.nn.functional.conv1d(padded_src_waveform[None, ...], ir_waveform[None, ...])[0]
    return convolved_speech_waveform


def process_spectrograms_from_file(audio_path, sample_rate, out_path):
    waveform, sr = torchaudio.load(audio_path)
    waveform = F.resample(waveform, sr, sample_rate)
    # crop file - as some short utterances, crops from 0.25s in to 1.25s
    waveform = waveform[0:1, 6000:sample_rate + 6000]
    spec = transform(waveform)
    torch.save(spec, out_path)

def process_spectrograms_from_object(waveform, sample_rate, out_path):
    # crop file - as some short utterances, crops from 0.25s in to 1.25s
    waveform = waveform[0:1, 6000:sample_rate + 6000]
    spec = transform(waveform)
    torch.save(spec, out_path)
print(len(train_files))

"""
From here below is the main generation loop for each category.
"""
# generate clean train files
print("Processing clean training files...")
count = 0
for file in train_files:
    count += 1
    process_spectrograms_from_file(file, sample_rate, spec_path / 'train/0.0/rt0.0_{:04}.pt'.format(count))

# generate clean test files
print("Processing clean test files...")
count = 0
for file in test_files:
    count += 1
    process_spectrograms_from_file(file, sample_rate, spec_path / 'test/rt0.0_{:04}.pt'.format(count))

# generate convolved test files
count = 0
print("Processing convolved test files...")
for ir in ir_filenames:
    count = 0
    print("Convolving files for {}s IR".format(ir.stem))
    processed_ir = process_ir(ir, sample_rate)
    for file in test_files:
        count += 1
        convolved_speech = process_speech(file, sample_rate, processed_ir)
        output_path = spec_path/'test'/'rt{}_{:04}.pt'.format(ir.stem, count)
        process_spectrograms_from_object(convolved_speech, sample_rate, output_path)

# generate convolved train files
count = 0
print("Processing convolved training files...")
for ir in ir_filenames[3:]:
    print("Convolving files for {}s IR".format(ir.stem))
    processed_ir = process_ir(ir, sample_rate)
    count = 0
    for file in train_files[3:]:
        count += 1
        convolved_speech = process_speech(file, sample_rate, processed_ir)
        output_path = spec_path/'train'/'{}'.format(ir.stem)/'rt{}_{:04}.pt'.format(ir.stem, count)
        process_spectrograms_from_object(convolved_speech, sample_rate, output_path)

