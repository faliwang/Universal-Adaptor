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
	wav_pth = '../wavs/tts'
	wav_dir = [
		'direct/hifigan', 'direct/waveglow', 'direct/wavernn',
		'w_adapt/hifigan_hifigan', 'w_adapt/hifigan_melgan', 'w_adapt/hifigan_waveglow', 'w_adapt/hifigan_wavernn',
		'w_adapt/waveglow_hifigan', 'w_adapt/waveglow_melgan', 'w_adapt/waveglow_waveglow', 'w_adapt/waveglow_wavernn',
		'w_adapt/wavernn_hifigan', 'w_adapt/wavernn_melgan', 'w_adapt/wavernn_waveglow', 'w_adapt/wavernn_wavernn'
	]
	sep = [28, 28, 28, 28, 28, 28, 28, 29]
	apis = [
		'https://script.google.com/macros/s/AKfycbzm3CRulLiwhY06royNegixP1kjVAgUDEsfZB8WDMAnIOxz4j_3oqeMAESOddAov-3X/exec',
		'https://script.google.com/macros/s/AKfycbxB_hyIi1COawQYc91BacaQ4KuLArwndO8Aa7KEVz-AYftMBRjpdUkcHQxq2LqMJrWIwA/exec',
		'https://script.google.com/macros/s/AKfycbwgWzztsB2QML2-6OKCfAS2N5s7XjRYtv51BQHeEgGJNpJjfNWiJn5d8ocTXzpCXL6Z/exec',
		'https://script.google.com/macros/s/AKfycbxPMmFlJed5XDByaT2_jn9jem66_iKdHWWPdz0zRKWeUEyhnhUaO2muNLkEzKih3sBC/exec',
		'https://script.google.com/macros/s/AKfycbxc4nmS4epzH_M6BGtGbRoFQjW-ZSFXomujo19vFWU4AyM5TxefmUfFr5H_xL64pHbllg/exec',
		'https://script.google.com/macros/s/AKfycbwcu3rjc54dRlnDUwlQ9T_yxYVz_R-FqTYYZ-2177EAp9HdcO0_WQBSeqSp99C7GeXwZQ/exec',
		'https://script.google.com/macros/s/AKfycbxh8g3ahMxThHMDd3uEoh8YCS9YzP-iu8qungspZ-BtjgWwkDDFo1NlIy8DJ-fV1BEh/exec',
		'https://script.google.com/macros/s/AKfycbyoCcilmorgfREiJSGCBpGpLaF9nG4mIu9ufggD3rXkedr3Pg8s50ybKsK8x6Mki_34hw/exec'
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
