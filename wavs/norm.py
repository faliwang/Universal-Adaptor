import os
import sys
import numpy as np
from scipy.io import wavfile


def load_wav(path):
	osr, wav = wavfile.read(path)
	assert osr == 22050
	wav = wav.astype(np.float32)
	wav = wav/np.max(np.abs(wav))
	return wav

def save_wav(wav, path):
	wav *= 32767 / max(0.01, np.max(np.abs(wav)))
	wavfile.write(path, 22050, wav.astype(np.int16))

pth = sys.argv[1]
for w in os.listdir(pth):
	if not w.endswith('.wav'):
		continue
	wpth = os.path.join(pth, w)
	wav = load_wav(wpth)
	os.remove(wpth)
	save_wav(wav, wpth)
