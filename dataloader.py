import tensorflow as tf
import os


def read_tfrecord(example):
    features = {
        "spec": tf.io.FixedLenFeature([16384], tf.float32),  # tf.string = bytestring (not text string)
        "f0": tf.io.FixedLenFeature([128], tf.float32),  # shape [] means scalar
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)
    spec = tf.reshape(example["spec"], (128, 128, 1))
    f0 = tf.reshape(example["f0"], (128,))
    return spec, f0


data_dir = "data/Hindi-English_test_utter-tfrec"
filenames = tf.io.gfile.glob(os.path.join(data_dir, "*.tfrec"))
testloader = tf.data.TFRecordDataset(filenames,
                                     num_parallel_reads=tf.data.AUTOTUNE)
testloader = testloader.map(read_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

filter_fn = lambda x, y: not tf.reduce_any(tf.math.is_nan(x)) and not tf.reduce_any(tf.math.is_nan(y))


real_data_dir = "data/Hindi-English_train_utter-tfrec"
filenames = tf.io.gfile.glob(os.path.join(real_data_dir, "*.tfrec"))
trainloader = tf.data.TFRecordDataset(filenames,
                                      num_parallel_reads=tf.data.AUTOTUNE)
trainloader = trainloader.map(read_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
trainloader = trainloader.filter(filter_fn)
