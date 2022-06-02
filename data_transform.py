# %%
"""VISUALIZE AND ANALYZE THE TRAINING DATA.
FUNCTIONS FOR TRANSFORMING THE INPUT DATA INTO THE REQUIRED FORMAT
FOR TRAINING."""

import librosa
from librosa import display
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.io.wavfile import write
import pyworld
from time import time, perf_counter, sleep
import tqdm
from joblib import Parallel, delayed

# %%

data_path = r"C:\Users\shast\CS-Synthesis\data\Hindi-English_train\Hin-Eng_utter_data"

filenames = os.listdir(data_path)

print("{} audio files found...".format(len(filenames)))


SLICE_LEN = 16384
SAMPLE_RATE = 16000  # over 1s for audio with sample rate 16kHz


# %%
# tensorflow data pipeline


def load_audio(fp, normalize=True):
    binary = tf.io.read_file(fp)
    audio, rate = tf.audio.decode_wav(binary)

    # To make the audio files as loud as possible and each independent audio file of same loudness
    if normalize:
        factor = tf.reduce_max(tf.abs(audio))
        if factor > 0:
            audio /= factor
    
    #get label
    # print(fp)
    # split_array = tf.strings.split(fp, sep="\\")
    # split_fname = tf.strings.split(split_array[-1], sep="_")
    # one_h_id = split_fname[0] == labels
    
    # label_id = tf.argmax(one_h_id)
    return tf.reshape(audio, shape=[-1, ]), rate


def frame_and_block(audio, rate, overlap_ratio=0., slice_len=16384):
    if overlap_ratio < 0:
        raise ValueError("Overlap ratio must be greater than zero")

    slice_hop = int(round(slice_len * (1. - overlap_ratio)) + 1e-4)

    if slice_hop < 1:
        raise ValueError("Overlap ratio too high")
    audio_slices = tf.signal.frame(tf.squeeze(audio, axis=1), frame_length=slice_len, frame_step=slice_hop,
                                   pad_value=0, pad_end=True)

    return tf.data.Dataset.from_tensor_slices(audio_slices), rate


# need to use min 2s frames to get voice samples

def crop_audio(audio, rate, crop_len=32768):
    diff = crop_len - tf.shape(audio)[0]
    if diff > 0:
        audio = tf.pad(audio, [[0, diff]])

    duration = tf.shape(audio)[0] // rate
    required_duration = crop_len // rate

    # if (duration - (required_duration + 1)) > 0:
    #     start = tf.experimental.numpy.random.randint(0, tf.cast(duration - (required_duration + 1), tf.int32))
    #     cropped = audio[start * rate: (start * rate) + crop_len]
    # else:
    cropped = audio[:crop_len]

    return cropped, rate


def trim_silences(audio, threshold=0.03):
    """
    trims the silences from the input audio
    :param audio: audio array
    :param threshold: silence threshold
    :return: trimmed audio
    """
    energy = librosa.feature.rms(audio.numpy())
    frames = np.nonzero(energy > threshold)
    ind = librosa.core.frames_to_samples(frames)
    indices = ind[1]

    return audio[indices[0]: indices[-1]] if indices.size else audio[0:0]


def tf_trim_silences(audio, threshold):
    suppressed = tf.py_function(trim_silences, inp=[audio, threshold], Tout=tf.float32)
    return suppressed


def filter_silences(x, y, threshold=0.7):
    """filter silences from a normalized spectrogram between [-1, 1]"""
    total_pixels = tf.reduce_prod(tf.shape(x))
    mask = tf.where(x < -0.5, 0.0, 1.0)
    number_ones = tf.reduce_sum(mask)
    ratio = number_ones / tf.cast(total_pixels, dtype=tf.float32)
    if ratio > threshold or tf.reduce_any(tf.math.is_nan(y)):
        out = False
    else:
        out = True

    return out


def mel_spectrogram(audio, frame_len, hop_len, fs, top_db=80.0, scale="db"):
    eps = 1e-10
    spec = tf.signal.stft(audio, frame_length=frame_len, frame_step=hop_len, pad_end=True)
    spec = tf.abs(spec)  # magnitude
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins=128,
                                                       num_spectrogram_bins=frame_len // 2 + 1,
                                                       sample_rate=fs,
                                                       upper_edge_hertz=tf.cast(fs, tf.float32) / 2,
                                                       )

    mel_spec = tf.matmul(spec ** 2, mel_matrix)
    mel_spec = tf.transpose(mel_spec)  # transpose from [frames, n_mel] -> [n_mel, frames]
    log_mel_spec = tf.math.log(tf.maximum(eps, mel_spec))
    db_mel_spec = 10 * tf.experimental.numpy.log10(tf.maximum(eps, mel_spec))
    db_mel_spec -= 10 * tf.experimental.numpy.log10(tf.maximum(eps, tf.reduce_max(mel_spec)))
    if top_db is not None:
        if top_db < 0:
            raise ValueError("top_db must be non-negative")
        db_mel_spec = tf.maximum(db_mel_spec, tf.reduce_max(db_mel_spec) - top_db)

    if scale == "db":
        out = tf.expand_dims(db_mel_spec, axis=-1)
    elif scale == "log":
        out = tf.expand_dims(log_mel_spec, axis=-1)
    elif scale == "linear":
        out = tf.expand_dims(mel_spec, axis=-1)
    else:
        raise NotImplementedError("not implemented")

    return out


# %%

"""x : ndarray
Input waveform signal.
fs : int
Sample rate of input signal in Hz.
frame_period : float
Period between consecutive frames in milliseconds. Default: 5.0
fft_size : int
Length of Fast Fourier Transform (in number of samples) The resulting dimension of ap adn sp will be fft_size // 2 + 1
"""

EPSILON = 1e-8


def save_fig(filename, figlist, log=True):
    # h = 10
    n = len(figlist)
    # peek into instances
    f = figlist[0]
    if len(f.shape) == 1:
        plt.figure()
        for i, f in enumerate(figlist):
            plt.subplot(n, 1, i + 1)
            if len(f.shape) == 1:
                plt.plot(f)
                plt.xlim([0, len(f)])
    elif len(f.shape) == 2:
        plt.figure()
        for i, f in enumerate(figlist):
            plt.subplot(n, 1, i + 1)
            if log:
                x = np.log(f + EPSILON)
            else:
                x = f + EPSILON
            plt.imshow(x.T, origin='lower', interpolation='none', aspect='auto', extent=(0, x.shape[0], 0, x.shape[1]))
    else:
        raise ValueError('Input dimension must < 3.')
    plt.savefig(filename)
    # plt.close()

# %%


# get tuple spectrogram and f0 as features for the model
def get_features(x, fs, frame_len=1024, hop_len=256, scale="db"):
    data = x.numpy().astype(np.float64)
    # frame_period = tf.cast(hop_len, tf.float32) * 1000. / tf.cast(fs, tf.float32)
    # _f0, t = pyworld.dio(data, tf.cast(fs, tf.int32), frame_period=frame_period)
    # f0 = pyworld.stonemask(data, _f0, t, tf.cast(fs, tf.int32))
    mel_spec = mel_spectrogram(x, frame_len, hop_len, fs, scale=scale)
    return mel_spec


def tf_get_features(x, fs, frame_len=1024, hop_len=256, scale="db"):
    x = tf.py_function(get_features,
                          [x, fs, frame_len, hop_len, scale],
                          [tf.float32, tf.float32])
    return x


# %%

"""
DATA NORMALIZATION
"""


# start = time()


# normalization_layer = tf.keras.layers.experimental.preprocessing.Normalization(axis=0)
# normalization_layer.adapt(genloader)  # this step takes more time ~800 sec
#
# normalized_instance = genloader.take(1).map(lambda x: normalization_layer(x))


# get normalization statistics of all the frequency bins
def get_norm_stats(instance):
    mean = np.mean(instance, axis=1)
    std = np.std(instance, axis=1)

    return mean, std


"""get stats of the data by iterating over the genloader by using joblib for parallel computing"""

# fp_dataset = tf.data.Dataset.list_files(os.path.join(data_path, "*.wav"), shuffle=False)
# audio_read = fp_dataset.map(lambda x: load_audio(x, normalize=True))
# genloader = audio_read.flat_map(lambda x: frame_and_block(x, overlap_ratio=0., SLICE_LEN=32768))
# genloader = genloader.map(lambda x: mel_spectrogram(x, 1024, 256, 16000., scale="log"),
#                       num_parallel_calls=tf.data.AUTOTUNE)
# genloader = genloader.filter(lambda x: filter_silences(x, threshold=0.7))
#
# results = Parallel(n_jobs=-1, verbose=1)(delayed(get_norm_stats)(i) for i in genloader.as_numpy_iterator())

# means = [mean for mean, _ in results]
# stds = [std for _, std in results]
#
# mel_means = np.array(means).mean(axis=0)
# mel_stds = np.array(stds).mean(axis=0)
# # save the data statistics in a .npz file for future reference
# np.savez("data_stats_log_spec.npz", means=mel_means, stds=mel_stds)

"""Normal iteration"""
# mel_means = []
# mel_stds = []
# for data in tqdm.tqdm(genloader.batch(16).as_numpy_iterator()):
#     means, stds = get_norm_stats(data)
#     mel_means.append(means)
#     mel_stds.append(stds)
#
# mel_means = np.array(mel_means).mean(axis=0)
# mel_stds = np.array(mel_stds).std(axis=0)
#
# print("Time taken for genloader adaption Normalization")
# print(time() - start)


# data-scale function to scale the data like in the Spec GAN paper (use when the spectrogram is log-scale not db-scale)

data_stats = np.load("data_stats_log_spec.npz")
mel_means = data_stats["means"].mean(axis=0)
mel_stds = data_stats["stds"].std(axis=0)


# normalizing the fundamental frequency between [0, 1] Min max scaler
def f0_norm(f0):
    # med = np.median(f0[f0 > 0])
    # f0[f0 == 0] = med
    log_f0 = 39.87 * tf.math.log((f0 / 50) + 1e-8)  # linear to log-semitone scale
    # log_f0 = f0
    min_f = tf.reduce_min(log_f0)
    max_f = tf.reduce_max(log_f0)
    norm_f0 = (log_f0 - min_f) / (max_f - min_f)
    return norm_f0


def normalize(x, y, X_mean, X_std):
    norm_X = (x - X_mean) / X_std
    norm_X /= 3.0
    norm_X = tf.clip_by_value(norm_X, -1.0, 1.0)
    if y is not None:
        norm_y = f0_norm(y)
    else:
        norm_y = y
    return norm_X, norm_y


def f_to_img(X_norm):  # from the wave-gan repo by chris donahue
    """

    :param X_norm: Normalised spectrogram
    :return: uint 8 image of the spectrogram
    """
    X_uint8 = X_norm + 1.
    X_uint8 *= 128.
    X_uint8 = tf.clip_by_value(X_uint8, 0., 255.)
    X_uint8 = tf.cast(X_uint8, tf.uint8)

    X_uint8 = tf.map_fn(lambda x: tf.image.rot90(x, 1), X_uint8)

    return X_uint8


# normalizer as in gan_synth [-1. , 1.]
def _range_normalizer(x, y, margin):
    x = tf.reshape(x, (-1,))
    min_x = tf.reduce_min(x)
    max_x = tf.reduce_max(x)
    a = margin * (2.0 / (max_x - min_x))
    b = margin * (-2.0 * min_x / (max_x - min_x) - 1.0)
    nr_x = tf.reshape(a * x + b, (128, 128, 1))
    if y is not None:
        norm_y = f0_norm(y)
    else:
        norm_y = y
    return tf.clip_by_value(nr_x, -1.0, 1.0), norm_y


def _range_denorm(x, margin):
    x = tf.reshape(x, (-1,))
    min_x = tf.reduce_min(x)
    max_x = tf.reduce_max(x)
    a = margin * (80.0 / (max_x - min_x))
    b = margin * (-80.0 * min_x / (max_x - min_x) - 80.0)
    de_x = tf.reshape(a * x + b, (128, 128, 1))

    return de_x


# %%

"""The data pipeline"""


def load_dataset(path, filter_sil=False, norm=None):
    fp_dataset = tf.data.Dataset.list_files(os.path.join(path, "*.wav"), shuffle=False)
    audio_read = fp_dataset.map(lambda x: load_audio(x, normalize=True), num_parallel_calls=tf.data.AUTOTUNE)

    # dataset = audio_read.interleave(lambda x, y: frame_and_block(x, overlap_ratio=0., SLICE_LEN=32768),
    #                                 cycle_length=1,
    #                                 num_parallel_calls=tf.data.AUTOTUNE)

    dataset = audio_read.map(lambda x, y: crop_audio(x, y, crop_len=16384), num_parallel_calls=tf.data.AUTOTUNE)

    # mel_dataset = dataset.map(lambda x: mel_spectrogram(x, 1024, 256, 16000., scale="db"),
    #                           num_parallel_calls=tf.data.AUTOTUNE)

    mel_dataset = dataset.map(lambda x, y: tf_get_features(x, y, 512, 128, scale="db"),
                              num_parallel_calls=tf.data.AUTOTUNE)

    if norm is not None:
        if norm == "range":
            norm_spec_data = mel_dataset.map(lambda x, y: _range_normalizer(x, y, margin=1.0),
                                             num_parallel_calls=tf.data.AUTOTUNE)

        elif norm == "standard":
            norm_spec_data = mel_dataset.map(lambda x, y: normalize(x, y, X_mean=mel_means, X_std=mel_stds),
                                             num_parallel_calls=tf.data.AUTOTUNE)

        else:
            raise NotImplementedError("Not implemented !")

        out_data = norm_spec_data

    else:
        out_data = mel_dataset

    if filter_sil:
        out_data = out_data.filter(lambda x, y: filter_silences(x, y, threshold=0.75))

    return out_data


# %%
# testing the data performance


def benchmark(dataset, num_epochs):
    start_time = perf_counter()
    for _ in range(num_epochs):
        for _ in tqdm.tqdm(dataset):
            sleep(0.01)
            # print("\b", end='')
            # print("=>", end="")

    print("Execution time: ", perf_counter() - start_time)


# %%
"""
CONVERT DATA INTO TFRecords format after all the preprocessing to load fast afterwards.
"""

"""Boiler plate code"""


def _bytestring_feature(list_of_bytestrings):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))


def _int_feature(list_of_ints):  # int64
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))


def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def to_tfrecord(spec, f0):
    """

    :param spec: a numpy ndarray of mel-spectrogram
    :param f0: a numpy ndarray of f0 contour
    :return: tf.train.Example to serialize and add to the record
    """
    feature = {
        'spec': _float_feature(spec.flatten().tolist()),  # audio is a list of floats
        'label': _int_feature(f0.flatten().tolist())
    }
    # Example is a flexible message type that contains key-value pairs,
    # where each key maps to a Feature message. Here, each Example contains
    # two features: A FloatList for the decoded audio data and an FloatList
    # containing the corresponding label's index.
    return tf.train.Example(features=tf.train.Features(feature=feature))


class TFRecordsWriter:
    def __init__(self, n_shards, split, out_dir, len_data):
        self.n_shards = n_shards
        self.split = split
        self.out_dir = out_dir
        self.len_data = len_data

    def _get_shard_path(self, split, shard_id, shard_size):
        return os.path.join(self.out_dir, f"{split}-{shard_id}-{shard_size}.tfrec")

    @staticmethod
    def _write_to_tfrecord(shard_path, shard_size, batched_examples):
        """

        :param shard_path: path of the single shard file
        :param shard_size: size of the examples in the shard
        :param batched_examples: sharded examples of the input genloader
        :return:  writes the tfrecord files in the specified output directory.
        """
        spec, f0 = batched_examples
        with tf.io.TFRecordWriter(shard_path) as writer:
            print("Writing {}...".format(shard_path))
            for i in range(shard_size):
                example = to_tfrecord(spec[i],
                                      f0[i])

                writer.write(example.SerializeToString())

    def convert(self, unbatched_dataset):
        """
        Converts the input genloader to tf records files in the initialized directory
        Call directly on the preprocessed genloader
        Method of the TFRecordWriter class.
        :param unbatched_dataset: tf genloader which you want to convert to tf records
        :return:
        """
        shard_size = np.ceil(1.0 * self.len_data / self.n_shards)
        sharded_data = unbatched_dataset.batch(shard_size)
        shard = 0
        for spec, f0 in sharded_data.as_numpy_iterator():
            print(".")
            shard_size = spec.shape[0]
            filename = self._get_shard_path(self.split, shard + 1, shard_size)
            self._write_to_tfrecord(filename, shard_size, (spec, f0))

            print("Wrote {} containing {} records".format(filename, shard_size))
            shard += 1


OUT_DIR = r"C:\Users\shast\CS-Synthesis\data\sc09_tfrecords"

if __name__ == "__main__":
    start = time()

    train_dataset = load_dataset(data_path, filter_sil=False, norm="range")

    unnorm_dataset = load_dataset(data_path, filter_sil=False)

    for (x, y) in train_dataset.shuffle(100).take(1).as_numpy_iterator():
        print(tf.shape(x), tf.shape(y))
        print(y)
        de_nrx = _range_denorm(x, margin=1.0)
        print("denormed_x: {}".format(de_nrx))
        fig, axs = plt.subplots(nrows=2, constrained_layout=True)
        X = np.linspace(1, x.shape[1], num=x.shape[1])
        Y = range(x.shape[0])
        img = axs[0].pcolormesh(X, Y, np.squeeze(x, axis=-1), shading="auto")
        axs[0].set_title("Mel-spectrogram")
        axs[0].set_xlabel("time-frames")
        axs[0].set_ylabel("Mel-frequency scale")
        fig.colorbar(img, ax=axs[0])
        axs[1].plot(y)
        axs[1].set_xlim([0, y.shape[0]])
        axs[1].set_title("Utterance")
        axs[1].set_xlabel("time-frames")
        axs[1].set_ylabel("Normalized frequency")

        fig.suptitle("Transformed input data", fontsize=16)
        fig.show()

    print("time taken to load the genloader")
    print(time() - start, "s")

    N_SHARDS = 16
    SPLIT = "train"
    # train_dataset = train_dataset.apply(tf.data.experimental.assert_cardinality(52817))
    len_data = len(train_dataset)
    # record_init = TFRecordsWriter(N_SHARDS,
    #                               SPLIT,
    #                               OUT_DIR,
    #                               len_data)
    # record_init.convert(train_dataset)

# %%

"""Read from tfrec_files"""


def read_tfrecord(example):
    features = {
        "spec": tf.io.FixedLenFeature([16384], tf.float32),  # tf.string = bytestring (not text string)
        "label": tf.io.FixedLenFeature([10], tf.int64),  # shape [] means scalar
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)
    spec = tf.reshape(example["spec"], (128, 128, 1))
    f0 = tf.reshape(example["label"], (10,))
    return spec, f0


if __name__ == "__main__":

    # option_no_order = tf.data.Options()
    # option_no_order.experimental_deterministic = False
    filenames = tf.io.gfile.glob(OUT_DIR + "/*.tfrec")
    tfrec_data = tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.AUTOTUNE)
    # tfrec_data = tfrec_data.with_options(option_no_order)
    tfrec_data = tfrec_data.map(read_tfrecord, num_parallel_calls=tf.data.AUTOTUNE).shuffle(300)

    for (x, y) in tfrec_data.take(1).as_numpy_iterator():
        print(tf.shape(x), tf.shape(y))
        fig, axs = plt.subplots(nrows=2, constrained_layout=True)
        X = np.linspace(0, x.shape[1], num=x.shape[1])
        Y = range(x.shape[0])
        img = axs[0].pcolormesh(X, Y, np.squeeze(x, axis=-1), shading="auto")
        axs[0].set_title("Mel-spectrogram")
        axs[0].set_xlabel("time-frames")
        axs[0].set_ylabel("Mel-frequency scale")
        fig.colorbar(img, ax=axs[0])
        axs[1].plot(y)
        axs[1].set_xlim([0, y.shape[0]])
        axs[1].set_title("f0 contour")
        axs[1].set_xlabel("time-frames")
        axs[1].set_ylabel("Normalized frequency")

        fig.suptitle("Data read from the TF-Records format", fontsize=16)
        fig.show()

# %%
