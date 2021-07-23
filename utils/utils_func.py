import argparse
import sys
import os

def get_files(path, extension='.wav'):
    files = os.listdir(path)
    wave_files = []
    for f in files:
        if f.endswith(extension):
            wave_files.append(f)
    wave_files.sort()
    return wave_files


def progbar(i, n, size=16):
    done = (i * size) // n
    bar = ''
    for i in range(size):
        bar += '█' if i <= done else '░'
    return bar


def stream(message):
    sys.stdout.write(f"\r{message}")


def valid_n_workers(num):
    n = int(num)
    if n < 1:
        raise argparse.ArgumentTypeError('%r must be an integer greater than 0' % num)
    return n


def check_config_valid(data1_config, wav_config=None, data2_config=None):
    if data1_config["sample_rate"] != data1_config["mel_config"]["sample_rate"]:
        raise ValueError("Sample rate is not equal! We got {} and {}".format(data1_config["sample_rate"], data1_config["mel_config"]["sample_rate"]))
    if data1_config["stft_config"]["n_fft"] != data1_config["mel_config"]["n_fft"]:
        raise ValueError("n_fft in stft & mel are not equal! We got {} and {}".format(data1_config["stft_config"]["n_fft"], data1_config["mel_config"]["n_fft"]))
    if data2_config != None:
        if data2_config["sample_rate"] != data2_config["mel_config"]["sample_rate"]:
            raise ValueError("Sample rate is not equal in data2! We got {} and {}".format(data2_config["sample_rate"], data2_config["mel_config"]["sample_rate"]))
        if data2_config["stft_config"]["n_fft"] != data2_config["mel_config"]["n_fft"]:
            raise ValueError("n_fft in stft & mel are not equal in data2! We got {} and {}".format(data2_config["stft_config"]["n_fft"], data2_config["mel_config"]["n_fft"]))
        if data1_config["sample_rate"] != data2_config["sample_rate"]:
            raise ValueError("Sample rate is not equal in data1 & 2! We got {} and {}".format(data1_config["sample_rate"], data2_config["sample_rate"]))
        if data1_config["stft_config"]["hop_length"] != wav_config["hop_length"]:
            raise ValueError("hop_length in data1 & wav are not equal! We got {} and {}".format(data1_config["stft_config"]["hop_length"], wav_config["hop_length"]))
        if data1_config["stft_config"]["win_length"] != wav_config["win_length"]:
            raise ValueError("win_length in data1 & wav are not equal! We got {} and {}".format(data1_config["stft_config"]["win_length"], wav_config["win_length"]))
    print("\nComplete checking config validity\n")