from tensorflow.keras.layers import Dense, Permute, Reshape, Input
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Permute, Reshape, Dropout
from tensorflow.keras.layers import Convolution2D, ReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import os
import tensorflow as tf
import pathlib
import tensorflow_io as tfio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyworld as pw
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from config import CONFIG
from dataloader import train_data, test_data, train_files
from sklearn.metrics import classification_report

NAME = "topcoder_crnn"
TRAIN_DATASET_PATH = "data/Language Identification/spectrograms cvit/train"
TEST_DATASET_PATH = "data/Language Identification/spectrograms cvit/test"

TRAIN_CSV_PATH = "train_gan_augmented.csv"
TEST_CSV_PATH = "test.csv"

train_df = pd.read_csv(TRAIN_CSV_PATH)
test_df = pd.read_csv(TEST_CSV_PATH)
train_df = shuffle(train_df)

AUTOTUNE = tf.data.AUTOTUNE

loaded = tf.saved_model.load("models/cs-synth-115k")
print(list(loaded.signatures.keys()))
augmenter = loaded.signatures["serving_default"]


def create_model(input_shape, config):
    weight_decay = 0.001

    model = Sequential()

    model.add(
        Convolution2D(32, (7, 7), kernel_regularizer=l2(weight_decay), activation="relu", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Convolution2D(64, (5, 5), kernel_regularizer=l2(weight_decay), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Convolution2D(128, (3, 3), kernel_regularizer=l2(weight_decay), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Convolution2D(256, (3, 3), kernel_regularizer=l2(weight_decay), activation="relu"))
    model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    bs, x, y, c = model.layers[-1].output_shape
    model.add(Reshape((x, y * c)))
    model.add(Bidirectional(LSTM(512, return_sequences=False)))
    model.add(Dense(64))
    model.add(Dropout(0.2))
    model.add(Dense(config["num_classes"], activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return model


def spec_aug(mel_spec, label):
    if tf.not_equal(label, 2):
        # time_warp = tfa.image.sparse_image_warp(mel_spec)
        freq_mask = tfio.audio.freq_mask(tf.squeeze(mel_spec), param=10)
        time_mask = tfio.audio.time_mask(freq_mask, param=10)

        return tf.expand_dims(time_mask, axis=-1), label
    else:
        return mel_spec, label


# data_init = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.2,
#                                                             rescale=1 / 255.0)
# train_ds = data_init.flow_from_directory(TRAIN_DATASET_PATH,
#                                          # x_col="train_images",
#                                          # y_col='class',
#                                          # directory=TRAIN_DATASET_PATH,
#                                          target_size=(128, 128),
#                                          color_mode="grayscale",
#                                          class_mode="sparse",
#                                          batch_size=CONFIG["batch_size"],
#                                          subset="training")
# val_ds = data_init.flow_from_directory(TRAIN_DATASET_PATH,
#                                        #   x_col="train_images",
#                                        #   y_col='class',
#                                        #   directory=TRAIN_DATASET_PATH,
#                                        target_size=(128, 128),
#                                        color_mode="grayscale",
#                                        class_mode="sparse",
#                                        batch_size=CONFIG["batch_size"],
#                                        subset="validation")

# test_data_init = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)
# test_ds = test_data_init.flow_from_dataframe(test_df,
#                                             x_col='test_images',
#                                             y_col='class',
#                                             target_size=(128, 128),
#                                             color_mode="grayscale",
#                                             class_mode="sparse",
#                                             batch_size=CONFIG["batch_size"],
#                                             shuffle=False
#                                             )

train_ds = train_data.skip(int(0.2 * len(train_files))).batch(CONFIG["batch_size"])
val_ds = train_data.take(int(0.2 * len(train_files))).batch(CONFIG["batch_size"])
test_ds = test_data


def kfold_cv(n_splits, train_df):
    best_auc = 0
    metrics = ["accuracy"]
    accs = []
    data_init = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0)
    kfold = KFold(n_splits=n_splits)

    for train_idx, val_idx in kfold.split(train_df):
        model = None
        model = create_model([128, 128, 1], CONFIG)

        train_split, val_split = train_df.iloc[train_idx], train_df.iloc[val_idx]
        print(train_split.head())
        print(val_split.head())

        train_ds = data_init.flow_from_dataframe(train_split,
                                                 x_col="train_images",
                                                 y_col='class',
                                                 # directory=TRAIN_DATASET_PATH,
                                                 target_size=(128, 128),
                                                 color_mode="grayscale",
                                                 class_mode="sparse",
                                                 batch_size=CONFIG["batch_size"],
                                                 # subset="training"
                                                 )
        val_ds = data_init.flow_from_dataframe(val_split,
                                               x_col="train_images",
                                               y_col='class',
                                               #   directory=TRAIN_DATASET_PATH,
                                               target_size=(128, 128),
                                               color_mode="grayscale",
                                               class_mode="sparse",
                                               batch_size=CONFIG["batch_size"],
                                               # subset="validation"
                                               )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=CONFIG["num_epochs"],
            callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=9),
        )

        scores = model.evaluate(test_ds)
        if scores[1] > best_auc:
            best_auc = scores[1]
            model.save("models/kfold/best_auc_model_unb.h5")

        accs.append(scores[1])

    return sum(accs) / len(accs)


if __name__ == "__main__":
    # kfold cross-validation
    # print(kfold_cv(5, train_df))    

    model = create_model([128, 128, 1], CONFIG)

    print(model.summary())

    # model_pth = "models_cvit/time_aug_imbalanced.h5"
    # model = tf.keras.models.load_model(model_pth)
    # print("loaded the model from {}".format(model_pth))

    history = model.fit(
        train_ds,
        validation_data=test_ds.batch(CONFIG["batch_size"]),
        epochs=CONFIG["num_epochs"],
        callbacks=[
            tf.keras.callbacks.EarlyStopping(verbose=1, patience=10),
            tf.keras.callbacks.ModelCheckpoint("models_cvit/ckpt", monitor="val_loss", save_best_only=True)]
    )

    metrics = history.history
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.show()

    model.save("models_/pitch_shift.h5")
