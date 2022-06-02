import runpy

import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from dataloader import trainloader
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import librosa
from data_transform import _range_denorm

train_data = trainloader.shuffle(1000)

# SpecAugment

time_mask_data = train_data.map(lambda x, y: tfio.audio.time_mask(tf.squeeze(x, axis=-1), param=10))
freq_mask_data = train_data.map(lambda x, y: tfio.audio.freq_mask(tf.squeeze(x, axis=-1), param=10))

for (x1, x2) in tf.data.Dataset.zip((time_mask_data, freq_mask_data)).take(1):
    fig, axs = plt.subplots(nrows=2,)
    print(x2.shape)
    X = librosa.frames_to_time(np.arange(x1.shape[0] + 1), sr=16000, hop_length=256)
    Y = np.arange(x1.shape[1] + 1)
    img1 = axs[0].pcolormesh(X, Y, x1, shading="auto")
    img2 = axs[1].pcolormesh(X, Y, x2, shading="auto")

    fig.show()

    spec = x1
    spec_denorm = _range_denorm(spec, margin=1.0)
    print("range: [{}, {}]".format(np.min(spec_denorm), np.max(spec_denorm)))
    pow_spec = librosa.db_to_power(np.reshape(spec_denorm, [128, 128]))
    aud = librosa.feature.inverse.mel_to_audio(pow_spec, 16000, n_fft=1024, hop_length=256, n_iter=60)
    print("Writing files to 'test/specsug-audio.wav'.....")
    write('test/specaug-audio.wav', rate=16000, data=aud)

