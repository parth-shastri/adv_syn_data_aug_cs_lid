"""

INFERENCE SCRIPT - perform Inference to generate synthetic samples using GANs

"""

import time
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from frechet_distance import calculate_fid, compute_embeddings, IMAGE_SIZE, BATCH_SIZE
from time import perf_counter
import math
import os
import numpy as np
import dataloader
from scipy.io.wavfile import write
from data_transform import _range_denorm


def _range_normalizer(x, limits=(-1., 1.), margin=1.0):
    x = tf.reshape(x, (-1,))
    min_x = tf.reduce_min(x)
    max_x = tf.reduce_max(x)
    a = margin * ((limits[1] - limits[0]) / (max_x - min_x))
    b = margin * (-(limits[1] - limits[0]) * min_x / (max_x - min_x) + limits[0])
    nr_x = tf.reshape(a * x + b, (128, 128, 1))
    return tf.clip_by_value(nr_x, limits[0], limits[1])


models_path = "models/saved models utterdata"
model_names = os.listdir(models_path)
print(model_names)
# model = tf.keras.models.load_model(os.path.join(models_path, "cs-synth-115k"))


if __name__ == "__main__":
    # load te train and test datasets for evaluation
    start = perf_counter()
    test_set = dataloader.testloader

    trainloader = dataloader.trainloader
    trainloader = trainloader.map(lambda x, y: _range_normalizer(x, limits=[0., 1.]), num_parallel_calls=tf.data.AUTOTUNE)
    # trainloader = trainloader.map(lambda x, y: tf.image.grayscale_to_rgb(x),
    #                               num_parallel_calls=tf.data.AUTOTUNE)

    # trainloader = trainloader.map(
    #     lambda x: tf.image.resize_with_pad(x, target_height=IMAGE_SIZE, target_width=IMAGE_SIZE),
    #     num_parallel_calls=tf.data.AUTOTUNE)

    # genloader = genloader.apply(tf.data.experimental.assert_cardinality(47412))
    trainloader = trainloader.shuffle(300).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    def fid_trend(testloader, trainloader, models_path):
        fid = list()
        for name in ["cs-synth-last-115k"]:
            in_loop_model = tf.keras.models.load_model(os.path.join(models_path, name))

            genloader = testloader.map(lambda x, y: in_loop_model(tf.expand_dims(y, axis=0), training=False),
                                     num_parallel_calls=tf.data.AUTOTUNE)
            genloader = genloader.map(lambda x: _range_normalizer(x, limits=[0., 1.]), num_parallel_calls=tf.data.AUTOTUNE)
            
            # genloader = genloader.map(lambda x: tf.image.grayscale_to_rgb(tf.squeeze(x, axis=0)),
            #                           num_parallel_calls=tf.data.AUTOTUNE)
            # genloader = genloader.map(
            #     lambda x: tf.image.resize_with_pad(x, target_height=IMAGE_SIZE, target_width=IMAGE_SIZE),
            #     num_parallel_calls=tf.data.AUTOTUNE)
            # genloader = genloader.apply(tf.data.experimental.assert_cardinality(47412))
            genloader = genloader.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

            count = math.ceil(2418 / BATCH_SIZE)

            # compute embeddings for real images
            real_image_embeddings = compute_embeddings(trainloader, count)

            # compute embeddings for generated images
            generated_image_embeddings = compute_embeddings(genloader, count)

            fid.append(calculate_fid(real_image_embeddings, generated_image_embeddings))
        print("time taken for execution: {:.3f}s".format(perf_counter() - start))

        return fid
    

    # fid = fid_trend(dataloader.testloader, trainloader, models_path)
    # fid = [1372.9548906235293, 204.63381235806105, 141.0344661552288, 156.17581703248965, 149.15755875581277]
    # fid = [881.6634785840863, 12.37959227993792, 8.530687272999803, 14.21280483299904, 9.665724242525837]

    # visualization
    n_examples = 4
    data_inp = test_set.shuffle(2500)

    ins = next(iter(data_inp.batch(n_examples)))
    model = tf.keras.models.load_model(os.path.join(models_path, "cs-synth-last-115k"))
    preds = model(ins[1], training=False)

    for i, (x, z, y) in enumerate(zip(preds, ins[0], ins[1])):
        start_time = time.perf_counter()

        fig, axs = plt.subplots(nrows=3, constrained_layout=True)
        X = librosa.frames_to_time(np.arange(x.shape[1] + 1), sr=16000, hop_length=256)
        Y = np.arange(x.shape[0] + 1)
        img = axs[0].pcolormesh(X, Y, np.squeeze(x, axis=-1), shading="auto", cmap="magma")
        axs[0].set_title("Generated Mel-spectrogram")
        axs[0].set_xlabel("time-frames")
        axs[0].set_ylabel("Mel-frequency scale")
        fig.colorbar(img, ax=axs[0])
        axs[1].plot(y)
        axs[1].set_xlim([0, y.shape[0]])
        axs[1].set_title("input f0 contour")
        axs[1].set_xlabel("time-frames")
        axs[1].set_ylabel("Normalized frequency")
        axs[2].pcolormesh(X, Y, np.squeeze(z, axis=-1), shading="auto", cmap="magma")
        axs[2].set_title("Mel-spectrogram")
        axs[2].set_xlabel("time-frames")
        axs[2].set_ylabel("Mel-frequency scale")

        fig.suptitle("Predictions", fontsize=16)
        fig.show()
        print("Inference time:".format(time.perf_counter() - start_time))

        # real audio re-sampled and generated using Griffin lim
        spec = np.squeeze(z, axis=-1)
        spec_denorm = _range_denorm(spec, margin=1.0)
        print("range: [{}, {}]".format(np.min(spec_denorm), np.max(spec_denorm)))
        pow_spec = librosa.db_to_power(np.reshape(spec_denorm, [128, 128]))
        aud = librosa.feature.inverse.mel_to_audio(pow_spec, 16000, n_fft=1024, hop_length=256, n_iter=60)
        print("Writing files to 'test/real_audio_{}.wav'.....".format(i))
        write('test/real_audio_{}.wav'.format(i), rate=16000, data=aud)

        # generated audio re-sampled and generated using Griffin lim
        spec = np.squeeze(x, axis=-1)
        spec_denorm = _range_denorm(spec, margin=1.0)
        print("range: [{}, {}]".format(np.min(spec_denorm), np.max(spec_denorm)))
        pow_spec = librosa.db_to_power(np.reshape(spec_denorm, [128, 128]))
        aud = librosa.feature.inverse.mel_to_audio(pow_spec, 16000, n_fft=1024, hop_length=256, n_iter=60)
        print("Writing files to 'test/gen_audio_{}.wav'.....".format(i))
        write('test/gen_audio_{}.wav'.format(i), rate=16000, data=aud)

# print(fid)

# plt.style.use("seaborn")
# fig, ax = plt.subplots(constrained_layout=True)
# fig.suptitle("FID trend", fontweight="bold", fontsize="x-large")
# ax.plot([0, 35000, 57000, 93000, 115000], fid)
# ax.set_xticks([0, 20000, 40000, 60000, 80000, 100000, 120000])
# ax.set_xticklabels([0, 20000, 40000, 60000, 80000, 100000, 120000], rotation=70)
# ax.scatter([0], [6.15], marker="*")
# ax.legend()
# ax.set_ylabel("Frechet inception distance", fontweight="bold", fontsize="large")
# ax.set_xlabel("steps ", fontweight="bold")
# fig.savefig("test/fid.png")
# fig.show()
