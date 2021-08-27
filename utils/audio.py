import librosa
import numpy as np
from scipy.io import wavfile


def load_wav(path, sample_rate):
    y, _ = librosa.load(path, sr=sample_rate)
    return y


def save_wav(y, path, sample_rate):
    y *= 32767/max(0.01, np.max(np.abs(y)))
    wavfile.write(path, sample_rate, y.astype(np.int16))


def trim_silence(y, wav_config):
    return librosa.effects.trim(
                y,
                top_db=wav_config["trim_silence_threshold_in_db"],
                frame_length=wav_config["trim_frame_size"],
                hop_length=wav_config["trim_hop_size"],
            )[0]


def stft(y, spec_config):
    y = np.pad(y, [(spec_config["left_pad"],spec_config["right_pad"])], mode=spec_config["pad_mode"])
    return librosa.stft(y, n_fft=spec_config["n_fft"],
                        hop_length=spec_config["hop_length"],
                        win_length=spec_config["win_length"],
                        window=spec_config["window"],
                        center=spec_config["center"])


def stft_power(S, stft_power):
    return np.power(S, stft_power)


def stft_depower(S, stft_power):
    return np.power(S, 1/stft_power)


def mel_basis(wav_config, spec_config):
    return librosa.filters.mel(
            sr=wav_config["sample_rate"], n_fft=spec_config["n_fft"],
            n_mels=spec_config["num_mels"], fmin=spec_config["fmin"],
            fmax=spec_config["fmax"])


def inv_mel_basis(mel_basis):
    return np.linalg.pinv(mel_basis)


def linear_to_mel(S, mel_basis):
    return np.einsum('ij,jk->ik', mel_basis, S)


def mel_to_linear(S, inv_mel_basis):
    inverse = np.einsum('ij,jk->ik', inv_mel_basis, S)
    return np.maximum(1e-10, inverse)


def amp_to_db(S, post_config):
    log_base = post_config["log_base"]
    log_factor = post_config["log_factor"]
    S = np.clip(S, a_min=1e-5, a_max=None)
    if log_base == "e":
        S = log_factor*np.log(S)
    elif log_base == 2:
        S = log_factor*np.log2(S)
    elif log_base == 10:
        S = log_factor*np.log10(S)
    else:
        raise NotImplementedError('no base {} in amp_to_db'.format(log_base))
    return S


def db_to_amp(S, post_config):
    log_base = post_config["log_base"]
    log_factor = post_config["log_factor"]
    S = S/log_factor
    if type(log_base) == int:
        S = np.power(log_base, S)
    elif log_base == 'e':
        S = np.exp(S)
    else:
        raise NotImplementedError('no base {} in db_to_amp'.format(log_base))
    return S


def normalize(S, post_config):
    ref_level_db = post_config["ref_level_db"]
    min_level_db = post_config["min_level_db"]
    return np.clip((S-ref_level_db-min_level_db)/-min_level_db, 0, 1)


def denormalize(S, post_config):
    ref_level_db = post_config["ref_level_db"]
    min_level_db = post_config["min_level_db"]
    return (np.clip(S, 0, 1)*-min_level_db)+min_level_db+ref_level_db


def stft_to_wav(S, spec_config, n_iter):
    audio = librosa.griffinlim(
            S, n_iter=n_iter, hop_length=spec_config["hop_length"],
            win_length=spec_config["win_length"], window=spec_config["window"], center=spec_config["center"])
    audio = audio[spec_config["left_pad"]:]
    if spec_config["right_pad"] > 0:
        audio = audio[: -1 * spec_config["right_pad"]]
    return audio
