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
	wav_pth = '../wavs/vctk'
	wav_dir = [
		'src', 'direct/hifigan', 'w_adapt/hifigan_hifigan', 'w_adapt/melgan_hifigan',
		'w_adapt/waveglow_hifigan', 'w_adapt/wavernn_hifigan'
	]
	sep = [23, 23, 22, 22]
	apis = [
		'https://script.google.com/macros/s/AKfycbynoxe5b_DJiDhnluS3uIH67fBM6NLK5omrV7-7SaQzZmJ_W5BmIZ4LUFRDN0-69x7XOQ/exec',
		'https://script.google.com/macros/s/AKfycbyRijM4MdTdGhLBo6TqZ3iyFVXB71wsmpdkdemhk_dU1KZwEK3L2y5HV32mRoDupl89/exec',
		'https://script.google.com/macros/s/AKfycbxO83oqihWTdfvqt2vZdT_1_CWz05LU7q3DUhN5CWpuZjeuf-8dx8ssxff_IwZsbtLkuQ/exec',
		'https://script.google.com/macros/s/AKfycbwcGu2nhzrSFlpG2ONygEO6jr6j2g5JN1Om-PKRkiE8m52MxE7EosFqA-Yyp2RZ3POlrA/exec'
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
