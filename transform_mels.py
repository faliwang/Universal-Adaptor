from multiprocessing import Pool, cpu_count
import argparse
import numpy as np
import soundfile as sf
import json
import os

from utils.utils_func import get_files, progbar, stream, valid_n_workers, check_config_valid
from utils.utils_mel import melspectrogram, gen_stft, stft_to_wav


def process_mels(mel_and_config):
    path, data1_config, wav_config, data2_config = mel_and_config
    mel_id = path.split('/')[-1]
    mel_id = mel_id.split('.')[0]
    mel1 = np.load(path)
    stft1 = gen_stft(mel1, data1_config)
    wav_hat = stft_to_wav(stft1, wav_config)
    if wav_config["save_wav"]:
        wav_path = os.path.join(wav_config["wav_path"], '{}.wav'.format(mel_id))
        sf.write(wav_path, wav_hat, data1_config["sample_rate"])
    mel2 = melspectrogram(wav_hat, data2_config)
    if data2_config["dtype"] == "float":
        mel2 = mel2.astype(np.float32)
    f_name = os.path.join(data2_config["output_path"], f'{mel_id}.npy')
    np.save(f_name, mel2, allow_pickle=False)
    return mel_id


def transform_mels(num_workers, data1_config, wav_config, data2_config, extension='.npy'):
    
    mels_path = data1_config["mels_path"]
    mel_files = get_files(mels_path, extension)

    print(f'\n{len(mel_files)} {extension[1:]} files found in "{mels_path}"\n')

    if len(mel_files) == 0:

        print('Please point mels_path in transform_config.json to your dataset,')
        print('or use the --path option.\n')

    else:
        mel_and_config = []
        for f in mel_files:
            wav = os.path.join(mels_path, f)
            mel_and_config.append((wav, data1_config, wav_config, data2_config))
        n_workers = max(1, num_workers)
        pool = Pool(processes=n_workers)

        for i, item_id in enumerate(pool.imap_unordered(process_mels, mel_and_config), 1):
            bar = progbar(i, len(mel_files))
            message = f'{bar} {i}/{len(mel_files)} '
            stream(message)

        print('\nCompleted!\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocessing for TTS and vocoder')
    parser.add_argument('--extension', '-e', metavar='EXT', default='.npy', help='file extension to search for in dataset folder')
    parser.add_argument('--num_workers', '-w', metavar='N', type=valid_n_workers, default=cpu_count()-1, help='The number of worker threads to use for preprocessing')
    parser.add_argument('--config_file', '-c', metavar='FILE', default='trans_config.json', help='The file to use for the hyperparameters')
    args = parser.parse_args()

    # Load config from json
    with open(args.config_file, 'r') as f:
        configs = json.load(f)
    data1_config = configs['data1_config']
    wav_config = configs['wav_config']
    data2_config = configs['data2_config']

    check_config_valid(data1_config, wav_config=wav_config, data2_config=data2_config)
    transform_mels(args.num_workers, data1_config, wav_config, data2_config, extension=args.extension)