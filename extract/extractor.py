import numpy as np
from utils import audio


class Extractor:
    def __init__(self, config):
        self.wav_config = config["wav_config"]
        self.spec_config = config["spec_config"]
        self.post_config = config["post_config"]
        self.github_repo = config["github_repo"]

    def load(self, path):
        y = audio.load_wav(path, self.wav_config["sample_rate"])
        if self.wav_config["normalize_loudness"] is not None:
            pass
        if self.wav_config["peak_norm"] is not None:
            y = self.wav_config["peak_norm"]*y/np.abs(y).max()
        if np.abs(y).max() > 1:
            y = y/np.abs(y).max()
        if self.wav_config["trim_silence_threshold_in_db"] is not None:
            pass
        return y

    def save(self, y, path):
        audio.save_wav(y, path, self.wav_config["sample_rate"])

    def convert(self, y):
        if self.spec_config["preemphasis"] is not None:
            pass
        D = audio.stft(y, self.spec_config)
        S = np.abs(D)
        S = audio.stft_power(S, self.spec_config["stft_power"])
        if self.spec_config["mel_spec"]:
            S = audio.linear_to_mel(S, self.wav_config, self.spec_config)
        if self.post_config["amp_to_db"]:
            S = audio.amp_to_db(S, self.post_config)
            if self.post_config["normalize_spec"]:
                S = audio.normalize(S, self.post_config)
        return S

    def inverse(self, S, n_iter=32):
        if self.post_config["amp_to_db"]:
            if self.post_config["normalize_spec"]:
                S = audio.denormalize(S, self.post_config)
            S = audio.db_to_amp(S, self.post_config)
        if self.spec_config["mel_spec"]:
            if not hasattr(self, 'inv_mel_basis'):
                self.inv_mel_basis = audio.inv_mel_basis(
                        self.wav_config, self.spec_config)
            S = audio.mel_to_linear(S, self.inv_mel_basis)
        S = audio.stft_depower(S, self.spec_config["stft_power"])
        y = audio.stft_to_wav(S, self.spec_config, n_iter)
        if self.spec_config["preemphasis"] is not None:
            pass
        return y
