import random

import torch
import torch.nn as nn
import torchaudio.transforms as transforms

from constants import F_MAX, F_MIN, HOP_SIZE, N_FFT, N_MELS, SAMPLE_RATE


class LogMelSpectrogram(nn.Module):
    def __init__(self):
        super().__init__()
        self.melspectrogram = transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_SIZE,
            f_min=F_MIN,
            f_max=F_MAX,
            n_mels=N_MELS,
            normalized=False)

    def forward(self, audio):
        # Alignment correction to match with piano roll.
        # When they convert the input into frames,
        # pretty_midi.get_piano_roll uses `ceil`,
        # but torchaudio.transforms.melspectrogram uses `round`.
        padded_audio = nn.functional.pad(audio, (HOP_SIZE // 2, 0), 'constant')
        mel = self.melspectrogram(padded_audio)[:, :, 1:]
        mel = mel.transpose(-1, -2)
        mel = torch.log(torch.clamp(mel, min=1e-9))
        return mel


class ConvStack(nn.Module):
    def __init__(self, n_mels, cnn_unit, fc_unit):
        super().__init__()

        # shape of input: (batch_size, 1 channel, frames, input_features)
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, cnn_unit, (3, 3), padding=1),
            nn.BatchNorm2d(cnn_unit),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(cnn_unit, cnn_unit, (3, 3), padding=1),
            nn.BatchNorm2d(cnn_unit),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(cnn_unit, cnn_unit * 2, (3, 3), padding=1),
            nn.BatchNorm2d(cnn_unit * 2),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((cnn_unit * 2) * (n_mels // 4), fc_unit),
            nn.Dropout(0.5))

    def forward(self, mel):
        x = mel.unsqueeze(1)
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class Transcriber(nn.Module):
    def __init__(self, cnn_unit, fc_unit):
        super().__init__()

        self.melspectrogram = LogMelSpectrogram()

        self.frame_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.frame_fc = nn.Linear(fc_unit, 88)

        self.onset_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.onset_fc = nn.Linear(fc_unit, 88)

    def forward(self, audio):
        mel = self.melspectrogram(audio)

        x = self.frame_conv_stack(mel)  # (B, T, C)
        frame_out = self.frame_fc(x)

        x = self.onset_conv_stack(mel)  # (B, T, C)
        onset_out = self.onset_fc(x)
        return frame_out, onset_out


class Transcriber_RNN(nn.Module):
    def __init__(self, cnn_unit, fc_unit):
        super().__init__()
        # Notice: Changing the initialization order may fail the tests.
        self.melspectrogram = LogMelSpectrogram()

        self.frame_lstm = nn.LSTM(N_MELS, 88, batch_first=True, num_layers=2, bidirectional=True)
        self.frame_fc = nn.Linear(88*2, 88)

        self.onset_lstm = nn.LSTM(N_MELS, 88, batch_first=True, num_layers=2, bidirectional=True)
        self.onset_fc = nn.Linear(88*2, 88)

    def forward(self, audio):
        # TODO: Question 1
        mel = self.melspectrogram(audio)

        x = self.frame_conv_stack(mel)
        frame_out = self.frame_fc(x)

        x = self.onset_conv_stack(mel)
        onset_out = self.onset_fc(x)

        return frame_out, onset_out


class Transcriber_CRNN(nn.Module):
    def __init__(self, cnn_unit, fc_unit):
        super().__init__()
        # Notice: Changing the initialization order may fail the tests.
        self.melspectrogram = LogMelSpectrogram()

        self.frame_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.frame_lstm = nn.LSTM(fc_unit, 88, batch_first=True, num_layers=2, bidirectional=True)
        self.frame_fc = nn.Linear(88*2, 88)

        self.onset_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.onset_lstm = nn.LSTM(fc_unit, 88, batch_first=True, num_layers=2, bidirectional=True)
        self.onset_fc = nn.Linear(88*2, 88)

    def forward(self, audio):
        # TODO: Question 2
        mel = self.melspectrogram(audio)

        x = self.frame_conv_stack(mel)
        x = self.frame_lstm(x)
        frame_out = self.frame_fc(x)

        x = self.onset_conv_stack(mel)
        x = self.onset_lstm(x)
        onset_out = self.onset_fc(x)

        return frame_out, onset_out


class Transcriber_ONF(nn.Module):
    def __init__(self, cnn_unit, fc_unit):
        super().__init__()
        # Notice: Changing the initialization order may fail the tests.
        self.melspectrogram = LogMelSpectrogram()

        self.frame_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.frame_fc = nn.Linear(fc_unit, 88)

        self.onset_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.onset_lstm = nn.LSTM(fc_unit, 88, batch_first=True, num_layers=2, bidirectional=True)
        self.onset_fc = nn.Linear(88*2, 88)

        self.combined_lstm = nn.LSTM(88*2, 88, batch_first=True, num_layers=2, bidirectional=True)
        self.combined_fc = nn.Linear(88*2, 88)

    def forward(self, audio):
        # TODO: Question 3
        mel = self.melspectrogram(audio)

        frame_x = self.frame_conv_stack(mel)
        frame_x = self.frame_fc(frame_x)

        onset_x = self.onset_conv_stack(mel)
        onset_x = self.onset_lstm(onset_x)
        onset_x = self.onset_fc(onset_x)
        onset_out = nn.Sigmoid()(onset_x)

        combined_x = torch.cat([frame_x.detach(), onset_x.detach()], 2)
        combined_x = self.combined_lstm(combined_x)
        combined_x = self.combined_fc(combined_x)
        frame_out = nn.Sigmoid()(combined_x)

        return frame_out, onset_out