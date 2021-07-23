import numpy as np
import librosa


def normalize_mel(S, min_level_db):
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)


def denormalize_mel(S, min_level_db):
    return (np.clip(S, 0, 1) * -min_level_db) + min_level_db


def amp_to_db(x, log_config):
    x = np.maximum(1e-5, x)
    if log_config["log_base"] == 10:
        x = np.log10(x)
    elif log_config["log_base"] == 'e':
        x = np.log(x)
    elif log_config["log_base"] == 2:
        x = np.log2(x)
    else:
        raise NotImplementedError('no base {} in amp_to_db'.format(log_config["log_base"]))
    return log_config["log_factor"] * x


def db_to_amp(x, log_config):
    x = x / log_config["log_factor"]
    if type(log_config["log_base"]) == int:
        x = np.power(log_config["log_base"], x)
    elif log_config["log_base"] == 'e':
        x = np.exp(x)
    else:
        raise NotImplementedError('no base {} in amp_to_db'.format(log_config["log_base"]))
    return x


def linear_to_mel(spectrogram, mel_config):
    return librosa.feature.melspectrogram(
        S=spectrogram, 
        sr=mel_config["sample_rate"], n_fft=mel_config["n_fft"], n_mels=mel_config["num_mels"], 
        fmin=mel_config["fmin"], fmax=mel_config["fmax"])


def mel_to_linear(spectrogram, mel_config, stft_power):
    return librosa.feature.inverse.mel_to_stft(
        spectrogram, 
        power=stft_power, sr=mel_config["sample_rate"], n_fft=mel_config["n_fft"], 
        fmin=mel_config["fmin"], fmax=mel_config["fmax"])


def stft_power(y, stft_power):
    if stft_power == 1:
        return np.abs(y)
    elif stft_power == 2:
        return np.abs(y) ** 2
    else:
        raise NotImplementedError('no power {} in stft_power'.format(stft_power))


def stft(y, stft_config):
    return librosa.stft(
        y=y,
        n_fft=stft_config["n_fft"], hop_length=stft_config["hop_length"], win_length=stft_config["win_length"])


def normalize_wav(wav, peak_norm):
    peak = np.abs(wav).max()
    if peak_norm:
        wav /= peak_norm
    elif peak > 1.0:
        wav /= peak
    return wav


def melspectrogram(y, data_config):
    y = normalize_wav(y, data_config["peak_norm"])
    D = stft(y, data_config["stft_config"])
    if data_config["stft_power"]:
        D = stft_power(D, data_config["stft_power"])
    if data_config["mel_spec"]:
        D = linear_to_mel(D, data_config["mel_config"])
    if data_config["amp_to_db"]:
        D = amp_to_db(D, data_config["log_config"])
    if data_config["normalize_spec"]:
        D = normalize_mel(D, data_config["min_level_db"])
    return D


def gen_stft(mel, data_config):
    if mel.ndim > 2:
        mel = np.squeeze(mel)
    if data_config["normalize_spec"]:
        mel = denormalize_mel(mel, data_config["min_level_db"])
    if data_config["amp_to_db"]:
        mel = db_to_amp(mel, data_config["log_config"])
    if data_config["mel_spec"]:
        mel = mel.astype(np.float32)
        mel = mel_to_linear(mel, data_config["mel_config"], data_config["stft_power"])
    return mel


def stft_to_wav(stft, wav_config):
    return librosa.griffinlim(
        stft, n_iter=wav_config["n_iter"],
        hop_length=wav_config["hop_length"], win_length=wav_config["win_length"])