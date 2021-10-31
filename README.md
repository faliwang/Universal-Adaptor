# Acoustic-feature-converter

Official repository for 'Universal Adaptor: Converting Mel-Spectrograms Between Different Configurations for Speech Synthesis'

# Installation

Install the requirements by pip:

    pip install -r requirements.txt

# How To Use

### Train Your Own Model

All the procedures are written in `run.sh`. All you have to do is to fill in the expected folder names and run the command:

    bash run.sh

### Inference with Pretrained Models

If you want to skip the training and inference directly, there are two steps to do.

1. We have to pass through Stage 1 by `transform.py`.

    python3 transform.py -sc source_config -tc target_config -d data_dir -o out_dir

2. We pass through Stage 2 by the stage 2 in `run.sh`. Fill in the expected folder names and run the command:

    bash run.sh

Then, you can find the results in your output directory.

# Samples

[Can be found here.](https://bogihsu.github.io/Acoustic-feature-converter/demo/demo.html)

____

### Reference Repositories for Configurations

* [Efficient Neural Audio Synthesis](https://github.com/fatchord/WaveRNN)
* [WaveGlow: A Flow-based Generative Network for Speech Synthesis](https://github.com/NVIDIA/waveglow)
* [MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis](https://github.com/descriptinc/melgan-neurips)
* [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://github.com/jik876/hifi-gan)
* [Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram](https://github.com/kan-bayashi/ParallelWaveGAN)
* [WaveNet: A generative model for raw audio](https://github.com/r9y9/wavenet_vocoder)