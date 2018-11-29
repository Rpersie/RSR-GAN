
# coding: utf-8

# In[13]:


from __future__ import print_function, division

import json
import numpy as np
import pandas as pd

import librosa
import soundfile as sf

import torch
from torch.utils.data import Dataset

from keras.preprocessing.sequence import pad_sequences
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


# In[5]:


class SpeechDataset(Dataset):
    """Speech dataset."""

    def __init__(self, csv_file, labels_file, audio_conf, transform=None, normalize=True):
        """
        Args:
            csv_file (string): Path to the csv file contain audio and transcript path.
            labels_file (string): Path to the json file contain label dictionary.
            audio_conf (dict) : Audio config info.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.speech_frame = pd.read_csv(csv_file, header=None)
        with open(labels_file, 'r') as f:
            self.labels = json.loads(f.read())
        self.window = audio_conf['window']
        self.window_size = audio_conf['window_size']
        self.window_stride = audio_conf['window_stride']
        self.sampling_rate = audio_conf['sampling_rate']
        self.transform = transform
        self.normalize = normalize

    def __len__(self):
        return len(self.speech_frame)

    def __getitem__(self, idx):
        wav_file = self.speech_frame.iloc[idx, 0]
        transcript_file = self.speech_frame.iloc[idx, 1]
        
        signal, _ = sf.read(wav_file)
        signal /= 1 << 31
        signal = self.spectrogram(signal)
        
        with open(transcript_file, 'r') as f:
            transcript = f.read().strip()
        transcript_idx = []
        transcript_idx.append(self.labels['<sos>'])
        for char in list(transcript):
            if char in self.labels:
                transcript_idx.append(self.labels[char])
        transcript_idx.append(self.labels['<eos>'])
        sample = {'signal': signal, 'transcript': np.array(transcript_idx)}
        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def spectrogram(self, signal):
        n_fft = int(self.sampling_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sampling_rate * self.window_stride)
        # STFT
        D = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length,
                        window=self.window, win_length=win_length)
        spect, phase = librosa.magphase(D)
        # S = log(S+1)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)
        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)
            
        return spect


# In[6]:


class Padding(object):
    """Rescale the audio signal and transcript to a given size.

    Args:
        signal_size (int): Desired output size of signal.
        transcript_size (int): Desired output size of transcript.
        labels_file (string): Path to the json file contain label dictionary.
    """

    def __init__(self, signal_size, transcript_size, labels_file):
        assert isinstance(signal_size, (int))
        assert isinstance(transcript_size, (int))
        self.signal_size = signal_size
        self.transcript_size = transcript_size
        with open(labels_file, 'r') as f:
            self.labels = json.loads(f.read())

    def __call__(self, sample):
        signal, transcript = sample['signal'], sample['transcript']
        signal /= 1 << 31
        signal = pad_sequences(signal, 
                               maxlen=self.signal_size, padding='post', 
                               truncating='post', value=0.0, dtype='float')
        transcript = pad_sequences(transcript.reshape(1, -1), 
                               maxlen=self.transcript_size, padding='post', 
                               truncating='post', value=self.labels['pad'], dtype='int')
        
        return {'signal': signal, 'transcript': transcript}


# In[7]:


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        signal, transcript = sample['signal'], sample['transcript']

        return {'signal': torch.from_numpy(signal),
                'transcript': torch.from_numpy(transcript)}

