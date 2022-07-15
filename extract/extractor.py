import numpy as np
from utils import audio


class Extractor:
    def __init__(self, config):
        self.wav_config = config["wav_config"]
        self.spec_config = config["spec_config"]
        self.post_config = config["post_config"]
        self.github_repo = config["github_repo"]
        self.mel_basis = audio.mel_basis(
                        self.wav_config, self.spec_config)
        self.inv_mel_basis = audio.inv_mel_basis(self.mel_basis)

    def load(self, path):
        y = audio.load_wav(path, self.wav_config["sample_rate"])
        if self.wav_config["normalize_loudness"] is not None:
            y = audio.normalize_loudness(y, self.wav_config["sample_rate"], self.wav_config["normalize_loudness"])
        if self.wav_config["peak_norm"] is not None:
            y = self.wav_config["peak_norm"]*y/np.abs(y).max()
        if np.abs(y).max() > 1:
            y = y/np.abs(y).max()
        if self.wav_config["trim_silence"]:
            y = audio.trim_silence(y, self.wav_config)
        return y

    def save(self, y, path):
        audio.save_wav(y, path, self.wav_config["sample_rate"])

    def convert(self, y):
        if self.wav_config["highpass_cutoff"] > 0.0:
            y = audio.low_cut_filter(y, self.wav_config["sample_rate"], self.wav_config["highpass_cutoff"])
        if self.spec_config["preemphasis"] is not None:
            y = audio.preemphasis(y, self.spec_config["preemphasis"])
        D = audio.stft(y, self.spec_config)
        S = np.abs(D)
        S = audio.stft_power(S, self.spec_config["stft_power"])
        if self.spec_config["mel_spec"]:
            S = audio.linear_to_mel(S, self.mel_basis)
        if self.post_config["amp_to_db"]:
            S = audio.amp_to_db(S, {"log_base": 'e', "log_factor": 1})
        return S

    def post_convert(self, S):
        if self.post_config["amp_to_db"]:
            S = audio.db_change_base(S, self.post_config)
            if self.post_config["normalize_spec"]:
                S = audio.normalize(S, self.post_config)
        return S

    def inverse(self, S, n_iter=32):
        if self.post_config["amp_to_db"]:
            if self.post_config["normalize_spec"]:
                S = audio.denormalize(S, self.post_config)
            S = audio.db_to_amp(S, self.post_config)
        if self.spec_config["mel_spec"]:
            S = audio.mel_to_linear(S, self.inv_mel_basis)
        S = audio.stft_depower(S, self.spec_config["stft_power"])
        y = audio.stft_to_wav(S, self.spec_config, n_iter)
        if self.spec_config["preemphasis"] is not None:
            y = audio.deemphasis(y, self.spec_config["preemphasis"])
        return y
    
    def post_inverse(self, S):
        if self.post_config["amp_to_db"]:
            if self.post_config["normalize_spec"]:
                S = audio.denormalize(S, self.post_config)
            S = audio.db_to_amp(S, self.post_config)
        return S
