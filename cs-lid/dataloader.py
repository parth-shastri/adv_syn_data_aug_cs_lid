from collections import namedtuple
import os
import tensorflow as tf
import pathlib
import tensorflow_io as tfio
import librosa
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import pyworld as pw
import pandas as pd
import matplotlib.pyplot as plt
from config import CONFIG
from tensorflow_addons.image import sparse_image_warp


TRAIN_DATASET_PATH = r"C:\Users\shast\OneDrive\Desktop\Language Identification\data\Language Identification\16 KHz\train"
TEST_DATASET_PATH = r"C:\Users\shast\OneDrive\Desktop\Language Identification\data\Language Identification\16 KHz\test"
SYNTH_DATA_PATH = r"C:\Users\shast\OneDrive\Desktop\Language Identification\data\synthetic data"

synth_filenames = os.listdir(SYNTH_DATA_PATH)
print(len(synth_filenames))

loaded = tf.saved_model.load(r"C:\Users\shast\OneDrive\Desktop\Language Identification\models\cs-synth-115k")
print(list(loaded.signatures.keys()))
augmenter = loaded.signatures["serving_default"]

AUTOTUNE = tf.data.AUTOTUNE
SAMPLE_RATE = 16000

data_dir = pathlib.Path(TRAIN_DATASET_PATH)

labels = np.array(tf.io.gfile.listdir(str(data_dir)))
labels = labels[labels != 'README.md']
print('labels:', labels)

filenames = glob.glob(os.path.join(TRAIN_DATASET_PATH, "**/*.wav"), recursive=True)
filenames = tf.random.shuffle(filenames, seed=133)
num_samples = len(filenames)
print('Number of total examples:', num_samples)

for command in labels:
    print('Number of examples in {}:'.format(command),
        len(glob.glob(os.path.join(str(data_dir / command), "**/*.wav"), recursive=True)))

train_files = np.array(filenames)
train_labels = [np.argmax(labels == tf.strings.split(input=name, sep=os.path.sep)[-2]) for name in train_files]
print(train_files[0], train_labels[0])
test_files = np.array(glob.glob(os.path.join(TEST_DATASET_PATH, "**/*.wav"), recursive=True))
test_labels = [np.argmax(labels == tf.strings.split(input=name, sep=os.path.sep)[-2]) for name in test_files]

# print(train_labels)
print('Training set size', len(train_files))
# print('Validation set size', len(val_files))
print('Test set size', len(test_files))


def load_img(path):
    img_binary = tf.io.read_file(path)
    img = tf.io.decode_png(img_binary, channels=1)
    img = tf.cast(img, tf.float32)
    img /= 255.0
    label = tf.cast(2, tf.int32)
    return img, label


def reduce_examples(files, label_array, label=2):
    mask = label == np.array(label_array)
    idx_list = []
    for i, item in enumerate(mask):
        if item:
            idx_list.append(i)
    np.random.seed(10)
    reduced_idxs = np.random.choice(idx_list, int(sum(mask)*0.2), replace=False)
    class_files, class_labels = files[reduced_idxs], np.array(label_array)[reduced_idxs]
    # print(len(reduced_idxs))
    temp_files, temp_labels = np.delete(files, idx_list), np.delete(label_array, idx_list)
    files, labels = np.concatenate([class_files, temp_files], axis=-1),  np.concatenate([class_labels, temp_labels], axis=-1)
    return files, labels


train_files, train_labels = reduce_examples(train_files, train_labels, label=2)
print("reduced train images:", len(train_files))


# utility function
def plot_spectrogram(spectrogram, ax):
    # Convert the frequencies to log scale and transpose, so that the time is
    # represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    
    if len(spectrogram.shape) > 2 and spectrogram.shape[2] != 3:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram)
        height = spectrogram.shape[0]
        width = spectrogram.shape[1]
        X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
        Y = range(height)
        ax.pcolormesh(X, Y, spectrogram, shading="auto")
    else:
        ax.imshow(tf.image.flip_up_down(spectrogram))


def decode_audio(audio_binary):
    # Decode WAV-encoded audio files to `float32` tensors, normalized
    # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.

    audio, rate = tf.audio.decode_wav(contents=audio_binary)
    rate_in = tf.cast(rate, tf.int64)

    # if rate_in != 16000:
    #     audio = tfio.audio.resample(audio, rate_in, 16000)
    # print(tf.shape(audio))
    # Since all the data is single channel (mono), drop the `channels`
    # axis from the array.
    audio = tf.squeeze(audio, axis=-1)
    
    return audio 


def get_label(file_path):
    parts = tf.strings.split(
        input=file_path,
        sep=os.path.sep)
    # Note: You'll use indexing here instead of tuple unpacking to enable this
    # to work in a TensorFlow graph.
    return parts[-2]


def trim_silences(audio, threshold=0.03):
    """
    trims the silences from the input audio
    :param audio: audio array
    :param threshold: silence threshold
    :return: trimmed audio
    """
    # try:
    energy = librosa.feature.rms(audio.numpy())
    frames = np.nonzero(energy > threshold)
    ind = librosa.core.frames_to_samples(frames)
    indices = ind[1]
    return audio[indices[0]: indices[-1]] if indices.size else audio[0:0]

    # except Exception as error:
    #     # print(audio.numpy())
    #     return audio[0:0]


def tf_trim_silences(audio, threshold):
    suppressed = tf.py_function(trim_silences, inp=[audio, threshold], Tout=tf.float32)
    return suppressed


def crop_audio(audio, label, rate, crop_len=32768):
    diff = crop_len - tf.shape(audio)[0]
    if diff > 0:
        audio = tf.pad(audio, [[0, diff]])

    duration = tf.shape(audio)[0] // rate
    required_duration = crop_len // rate

    # if (duration - (required_duration + 1)) > 0:
    #     start = tf.experimental.numpy.random.randint(0, duration - (required_duration + 1), dtype=tf.int32)
    #     cropped = audio[start * rate: (start * rate) + crop_len]
    # else:
    cropped = audio[:crop_len]

    return cropped, label


def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    waveform = tf_trim_silences(waveform, 0.03)
    # waveform = crop_audio(waveform, SAMPLE_RATE, crop_len=65536)
    return waveform, label


def filter_small(audio, label):
    if tf.shape(audio)[0] < 32768:
        return False
    else:
        return True


def mel_spectrogram(audio, label, frame_len, hop_len, fs, top_db=80.0, scale="db", aug=False):
    eps = 1e-10
    spec = tf.signal.stft(audio, frame_length=frame_len, frame_step=hop_len, pad_end=True)
    spec = tf.abs(spec)  # magnitude
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins=128,
                                                       num_spectrogram_bins=frame_len // 2 + 1,
                                                       sample_rate=fs,
                                                       upper_edge_hertz=fs / 2,
                                                       )

    mel_spec = tf.matmul(spec ** 2, mel_matrix)
    if aug:
        mel_spec = specaug(mel_spec, label)
        # print(tf.shape(mel_spec))
    # mel_spec = tf.transpose(mel_spec)  # transpose from [frames, n_mel] -> [n_mel, frames]
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

    return tf.transpose(out, perm=[1, 0, 2])


def f0_norm(f0):
    # med = np.median(f0[f0 > 0])
    # f0[f0 == 0] = med
    log_f0 = 39.87 * tf.math.log((f0 / 50) + 1e-8)  # linear to log-semitone scale
    # log_f0 = f0
    min_f = tf.reduce_min(log_f0)
    max_f = tf.reduce_max(log_f0)
    norm_f0 = (log_f0 - min_f) / (max_f - min_f)
    return norm_f0


def sparse_warp(mel_spectrogram, time_warping_para=50):
    """Spec augmentation Calculation Function.
    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.
    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to warping and masking.
      time_warping_para(float): Augmentation parameter, "time warp parameter W".
        If none, default = 80 for LibriSpeech.
    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """

    fbank_size = tf.shape(mel_spectrogram)
    n, v = fbank_size[0], fbank_size[1]

    # Step 1 : Time warping
    # Image warping control point setting.
    # Source
    pt = tf.random.uniform([], time_warping_para, n-time_warping_para, tf.int32) # radnom point along the time axis
    src_ctr_pt_freq = tf.range(v // 2)  # control points on freq-axis
    src_ctr_pt_time = tf.ones_like(src_ctr_pt_freq) * pt  # control points on time-axis
    src_ctr_pts = tf.stack((src_ctr_pt_time, src_ctr_pt_freq), -1)
    src_ctr_pts = tf.cast(src_ctr_pts, dtype=tf.float32)

    # Destination
    w = tf.random.uniform([], -time_warping_para, time_warping_para, tf.int32)  # distance
    dest_ctr_pt_freq = src_ctr_pt_freq
    dest_ctr_pt_time = src_ctr_pt_time + w
    dest_ctr_pts = tf.stack((dest_ctr_pt_time, dest_ctr_pt_freq), -1)
    dest_ctr_pts = tf.cast(dest_ctr_pts, dtype=tf.float32)

    # warp
    source_control_point_locations = tf.expand_dims(src_ctr_pts, 0)  # (1, v//2, 2)
    dest_control_point_locations = tf.expand_dims(dest_ctr_pts, 0)  # (1, v//2, 2)

    warped_image, _ = sparse_image_warp(mel_spectrogram,
                                        source_control_point_locations,
                                        dest_control_point_locations)
    return warped_image


def specaug(mel_spec, label):
    # Augmentation factor 2x    
    if tf.equal(label, 2) and tf.random.uniform(()) > 0.2:
        # time_warp = tfa.image.sparse_image_warp(mel_spec)
        # warped_spec = sparse_warp(mel_spec, time_warping_para=50)
        freq_mask = tfio.audio.freq_mask(mel_spec, param=13)
        time_mask = tfio.audio.time_mask(freq_mask, param=20)

        return time_mask
    else:
        return mel_spec


def gan_augment(waveform, label):
    data = np.array(waveform).astype(np.float64)
    frame_period = tf.cast(256., tf.float32) * 1000. / tf.cast(SAMPLE_RATE, tf.float32)
    # f0, _, _ = pw.wav2world(data, SAMPLE_RATE, frame_period, 1024)
    _f0, t = pw.dio(data, SAMPLE_RATE, frame_period=frame_period)
    f0 = pw.stonemask(data, _f0, t, SAMPLE_RATE)
    f0 = f0_norm(f0[:128])
    with tf.device("CPU:0"):
        pred = augmenter(input_1=tf.cast(tf.expand_dims(f0, axis=0), tf.float32))
        # print(pred)
    out = pred["output_1"][0]

    if pred is None:
        pred = mel_spectrogram(waveform, label, 1024, 256, SAMPLE_RATE)
        out = _range_normalizer(pred)

    return out


def time_stretch(audio):
    input_length = 32768
    rate = np.random.uniform(0.5, 1.5)
    timeshifted_audio = librosa.effects.time_stretch(np.array(audio), rate)
    if len(timeshifted_audio) > input_length:
        timeshifted_audio = timeshifted_audio[:input_length]
    else:
        timeshifted_audio = np.pad(timeshifted_audio, (0, max(0, input_length - len(timeshifted_audio))), "constant")
    return timeshifted_audio


def tf_time_stretch(audio, label):
    if label == "Hindi-English" and tf.random.uniform(()) > 0.2:
        timeshift_audio = tf.py_function(time_stretch, [audio], [tf.float32])
        timeshift_audio = timeshift_audio[0]
    else:
        timeshift_audio = audio
    return timeshift_audio, label


def pitch_shift(audio):
    n_steps = np.random.randint( -4, 4)
    pitchshifted_audio = librosa.effects.pitch_shift(np.array(audio), 16000, n_steps)
    return pitchshifted_audio


def tf_pitchshift(audio, label):
    if label == "Hindi-English" and tf.random.uniform(()) > 0.2:
        pitchshifted_audio = tf.py_function(pitch_shift, [audio], [tf.float32])
        pitchshifted_audio = pitchshifted_audio[0]
    else:
        pitchshifted_audio = audio
    
    return pitchshifted_audio, label


def clean_bad_spec(spec, label):
    if tf.math.count_nonzero(spec - tf.reduce_mean(spec)) == 0:
        out = False
    else:
        out = True
    
    return out


def filter_silences(x, y, threshold=0.9):
    """filter silences from a normalized spectrogram between [0, 1]"""
    total_pixels = tf.reduce_prod(tf.shape(x))
    mask = tf.where(x < 0.02, 0.0, 1.0)
    number_ones = tf.reduce_sum(mask)
    ratio = number_ones / tf.cast(total_pixels, dtype=tf.float32)
    if ratio < threshold:
        out = False
    else:
        out = True

    return out


def get_spectrogram_and_label_id(audio, label, augment=False, spec_aug=False):
    label_id = np.argmax(label == labels)
    if augment and tf.random.uniform((), seed=1332) > 0.2 and label_id == 2:
        spectrogram = gan_augment(audio, label_id)
    else:
        spectrogram = mel_spectrogram(audio, label_id, frame_len=1024, hop_len=256, fs=SAMPLE_RATE, aug=spec_aug)
        spectrogram = _range_normalizer(spectrogram, limits=[-1.0, 1.0])

    return spectrogram, label_id


def tf_get_spectrogram_and_label(audio, label, augment=False, spec_aug=False):
    spec, label_id = tf.py_function(get_spectrogram_and_label_id, [audio, label, augment, spec_aug], [tf.float32, tf.int32])
    return spec, label_id


def min_max_normalizer(x):
    x = tf.reshape(x, (-1,))
    min_x = tf.reduce_min(x)
    max_x = tf.reduce_max(x)
    nr_x = (x - min_x) / (max_x - min_x)
    nr_x = tf.reshape(nr_x, (128, 128, 1))
    return tf.clip_by_value(nr_x, 0.0, 1.0)


def _range_normalizer(x, limits=(-1., 1.), margin=1.0):
    x = tf.reshape(x, (-1,))
    min_x = tf.reduce_min(x)
    max_x = tf.reduce_max(x)
    a = margin * ((limits[1] - limits[0]) / (max_x - min_x))
    b = margin * (-(limits[1] - limits[0]) * min_x / (max_x - min_x) + limits[0])
    nr_x = tf.reshape(a * x + b, (128, 128, 1))
    return tf.clip_by_value(nr_x, limits[0], limits[1])


def preprocess_dataset(files, split="train", aug=False, spec_aug=False, trad_aug=None):
    rand_files = tf.random.shuffle(files, seed=1331)
    files_ds = tf.data.Dataset.from_tensor_slices(rand_files)
    output_ds = files_ds.map(
        map_func=get_waveform_and_label,
        num_parallel_calls=AUTOTUNE)
    
    output_ds = output_ds.filter(lambda x, y: filter_small(x, y))
    
    output_ds = output_ds.map(lambda x, y: crop_audio(x, y, SAMPLE_RATE, 32768))

    if trad_aug == "time stretch":
        print("time stretching...")
        output_ds = output_ds.map(tf_time_stretch, num_parallel_calls=AUTOTUNE)
    if trad_aug == "pitch shift":

        print("pitch shifting...")
        output_ds = output_ds.map(tf_pitchshift, num_parallel_calls=AUTOTUNE)

    if split == "train" and spec_aug:
        output_ds = output_ds.map(
            lambda x, y: tf_get_spectrogram_and_label(x, y, augment=False, spec_aug=True),
            num_parallel_calls=AUTOTUNE)
    else:
        output_ds = output_ds.map(
            lambda x, y: tf_get_spectrogram_and_label(x, y, augment=False),
            num_parallel_calls=AUTOTUNE)
    # output_ds = output_ds.map(lambda x, y: (_range_normalizer(x, limits=[-1., 1.]), y), num_parallel_calls=AUTOTUNE)

    output_ds = output_ds.filter(lambda x, y: not tf.reduce_any(tf.math.is_nan(x)))

    output_ds = output_ds.map(lambda x, y: (tf.reshape(x, [128, 128, 1]), tf.reshape(y, [])),
                              num_parallel_calls=tf.data.AUTOTUNE)
    output_ds = output_ds.map(lambda x, y: (_range_normalizer(x, limits=[0., 1.]), y), num_parallel_calls=AUTOTUNE)
    
    if split == "train" and aug:
        gen_data = tf.data.Dataset.list_files(os.path.join(SYNTH_DATA_PATH, "*.png"))
        gen_data = gen_data.map(load_img, num_parallel_calls=AUTOTUNE)
        output_ds = output_ds.concatenate(gen_data)

    output_ds = output_ds.map(lambda x, y: (tf.reshape(x, [128, 128, 1]), tf.reshape(y, [])),
                              num_parallel_calls=tf.data.AUTOTUNE)
    # output_ds = output_ds.filter(lambda x, y: filter_silences(x, y, threshold=0.6))
    # output_ds = output_ds.filter(lambda x, y: clean_bad_spec(x, y))

    # output_ds = output_ds.map(lambda x, y: (tf.image.grayscale_to_rgb(x), y),
    #                           num_parallel_calls=tf.data.AUTOTUNE)
    return output_ds


train_data = preprocess_dataset(train_files, aug=False, spec_aug=True, trad_aug="pitch shift")
print(train_data.element_spec)
test_data = preprocess_dataset(test_files, split="test")

IMAGE_DIR = "data/Language Identification/spectrograms cvit/train"
IMAGE_DIR_TEST = "data/Language Identification/spectrograms cvit/test"
labels = ["English", "Hindi", "Hindi-English"]

if __name__ == "__main__":

    rows = 3
    cols = 3
    n = rows * cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    cn = 0
    for i, (spectrogram, label_id) in enumerate(train_data):
        print(".")
        print(label_id)
        if label_id == 2:
            # label_id = np.argmax(label_id)
            r = cn // cols
            c = cn % cols
            cn += 1
            ax = axes[r][c]
            print(spectrogram.shape)
            print(np.min(spectrogram), np.max(spectrogram))
            plot_spectrogram(spectrogram.numpy(), ax)

            ax.set_title(f"{label_id}-{labels[label_id]}")
            ax.axis('off')
        
        if cn == n:
            break

    plt.show()
    
    try:
        os.mkdir(IMAGE_DIR)
    except FileExistsError:
        print("the train file already exists!!")

    for label in labels:
        try:
            os.mkdir(os.path.join(IMAGE_DIR, label))
        except FileExistsError:
            print("the train file already exists")
    try:
        os.mkdir(IMAGE_DIR_TEST)
    except FileExistsError:
        print("the test file already exists!!")

    for label in labels:
        try:
            os.mkdir(os.path.join(IMAGE_DIR_TEST, label))
        except FileExistsError:
            print("the test file already exists")

    if len(os.listdir(os.path.join(IMAGE_DIR, labels[0]))) == 0:
        for i, (image, label) in enumerate(tqdm(train_data)):
            im = Image.fromarray(np.reshape(np.uint8(image*255), (128, 128)))
            try:
                im.save(os.path.join(IMAGE_DIR, labels[label.numpy()], f"{labels[label.numpy()]}_{i}.png"))
            except Exception as error:
                print("{} error!".format(error))

    else:
        print("the files already exist!!")

    if len(os.listdir(os.path.join(IMAGE_DIR_TEST, labels[0]))) == 0:
        for i, (image, label) in enumerate(tqdm(test_data)):
            im = Image.fromarray(np.reshape(np.uint8(image*255), (128, 128)))
            try:
                im.save(os.path.join(IMAGE_DIR_TEST, labels[label.numpy()], f"{labels[label.numpy()]}_{i}.png"))
            except Exception as error:
                print("{} error!".format(error))
        
    else:
        print("the files already exist!!")

    train_img_filenames = []
    train_img_labels = []
    train_class = []
    for i, name in enumerate(labels):
        for img_name in os.listdir(os.path.join(IMAGE_DIR, name)):
            train_img_filenames.append(os.path.join(IMAGE_DIR, name, img_name))
            train_img_labels.append(i)
            train_class.append(name)

    test_img_filenames = []
    test_img_labels = []
    test_classes = []
    for i, name in enumerate(labels):
        for img_name in os.listdir(os.path.join(IMAGE_DIR_TEST, name)):
            test_img_filenames.append(os.path.join(IMAGE_DIR_TEST, name, img_name))
            test_img_labels.append(i)
            test_classes.append(name)

    dic = {"train_images": train_img_filenames, "label": train_img_labels, "class": train_class}
    train_df = pd.DataFrame(dic)
    train_df.to_csv("train_gan_augmented.csv")
    print(len(train_df), "train images in the dataframe...")

    dic = {"test_images": test_img_filenames, "label": test_img_labels, "class": test_classes}
    test_df = pd.DataFrame(dic)
    test_df.to_csv("test.csv")
    print(len(test_df), "test images in the dataframe...")
    # img = tf.io.read_file(r"C:\Users\shast\OneDrive\Desktop\Language Identification\data\Language Identification\spectrograms-50%\train\Hindi-English\Hindi-English_8809.png")
    # img = tf.io.decode_png(img)
    # print(tf.shape(img))
    # print(img)
    # img = tf.cast(img/255, tf.float32)

    # print(filter_silences(img, None, threshold=0.))
    # for data, label in train_data:
    #     print(tf.reduce_mean(data))
    #     break
