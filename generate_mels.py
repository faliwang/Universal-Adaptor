from multiprocessing import Pool, cpu_count
import argparse
import numpy as np
import librosa
import json
import os

from utils.utils_func import get_files, progbar, stream, valid_n_workers, check_config_valid
from utils.utils_mel import melspectrogram


def process_wav(wav_and_config):
    path, data_config = wav_and_config
    wav_id = path.split('/')[-1]
    wav_id = wav_id.split('.')[0]
    y = librosa.load(path, sr=data_config["sample_rate"])[0]
    mel = melspectrogram(y, data_config)
    if data_config["dtype"] == "float":
        mel = mel.astype(np.float32)
    f_name = os.path.join(data_config["data_path"], f'{wav_id}.npy')
    np.save(f_name, mel, allow_pickle=False)
    return wav_id


def generate_mels(extension, num_workers, data_config):
    
    wav_path = data_config["wav_path"]
    wav_files = get_files(wav_path, extension)

    print(f'\n{len(wav_files)} {extension[1:]} files found in "{wav_path}"\n')

    if len(wav_files) == 0:

        print('Please point wav_path in generate_config.json to your dataset,')
        print('or use the --path option.\n')

    else:
        wav_and_config = []
        for f in wav_files:
            wav = os.path.join(wav_path, f)
            wav_and_config.append((wav, data_config))
        n_workers = max(1, num_workers)
        pool = Pool(processes=n_workers)

        for i, item_id in enumerate(pool.imap_unordered(process_wav, wav_and_config), 1):
            bar = progbar(i, len(wav_files))
            message = f'{bar} {i}/{len(wav_files)} '
            stream(message)

        print('\nCompleted!\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocessing for TTS and vocoder')
    parser.add_argument('--extension', '-e', metavar='EXT', default='.wav', help='file extension to search for in dataset folder')
    parser.add_argument('--num_workers', '-w', metavar='N', type=valid_n_workers, default=cpu_count()-1, help='The number of worker threads to use for preprocessing')
    parser.add_argument('--config_file', '-c', metavar='FILE', default='gen_config.json', help='The file to use for the hyperparameters')
    args = parser.parse_args()

    # Load config from json
    with open(args.config_file, 'r') as f:
        configs = json.load(f)
    data_config = configs['data_config']

    check_config_valid(data_config)
    generate_mels(args.extension, args.num_workers, data_config)