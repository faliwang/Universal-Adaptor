import os
import random
import numpy as np
from jinja2 import Environment
from jinja2 import FileSystemLoader
random.seed(0)


def render(i, title, url, q):
	loader = FileSystemLoader(searchpath = './templates')
	env = Environment(loader = loader)
	template = env.get_template('mos.html.jinja2')

	html = template.render(
		page_title = title,
		form_url = url,
		form_id = 1,
		questions = q
	)
	with open('form-'+str(i+1)+'.html', 'w') as w:
		w.write(html)


if __name__ == '__main__':
	wav_pth = '../wavs/lj'
	wav_dir = [
		'src', 'direct/hifigan', 'direct/melgan', 'direct/waveglow', 'direct/wavernn',
		'w_adapt/hifigan_hifigan', 'w_adapt/hifigan_melgan', 'w_adapt/hifigan_waveglow', 'w_adapt/hifigan_wavernn',
		'w_adapt/melgan_hifigan', 'w_adapt/melgan_melgan', 'w_adapt/melgan_waveglow', 'w_adapt/melgan_wavernn',
		'w_adapt/waveglow_hifigan', 'w_adapt/waveglow_melgan', 'w_adapt/waveglow_waveglow', 'w_adapt/waveglow_wavernn',
		'w_adapt/wavernn_hifigan', 'w_adapt/wavernn_melgan', 'w_adapt/wavernn_waveglow', 'w_adapt/wavernn_wavernn'
	]
	sep = [31, 31, 31, 31, 31, 32, 32, 32, 32, 32]
	apis = [
		'',
		'',
		'',
		'',
		'',
		'',
		'',
		'',
		'',
		''
	]

	wav_list = []
	for d in wav_dir:
		wavs = sorted(
			os.listdir(os.path.join(wav_pth, d)))[:15]
		for x in wavs:
			wav_list.append(os.path.join(wav_pth, d, x))
	assert len(wav_list) == sum(sep)

	wav_list = sorted(wav_list)
	random.shuffle(wav_list)
	with open('shuffle.txt', 'w') as w:
		w.write('\n'.join(['/'.join(x.split('/')[3:]) for x in wav_list]))
	with open('sheet.tsv', 'w') as w:
		c = 0
		for x in sep:
			for i in range(x):
				w.write('/'.join(wav_list[c].split('/')[3:-1]))
				c += 1
				if i != x-1:
					w.write('\t')
			w.write('\n')

	qs=[]
	for i, x in enumerate(wav_list):
		qs.append(
			{
				'title': '問題 '+str(i+1),
				'audio_path': x,
				'name': 'q'+str(i+1)
			}
		)
	for i, x in enumerate(sep):
		s = 0
		for j in range(i):
			s += sep[j]
		e = s+x
		render(i, 'MOS 實驗表單 '+str(i+1), apis[i], qs[s:e])
