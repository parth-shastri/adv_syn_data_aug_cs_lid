"""COMPUTE THE FID FOR GAN EVALUATION"""
from ast import In
import tensorflow as tf
import numpy as np
import math
from tensorflow.keras.applications import InceptionV3
from tqdm import tqdm
import dataloader
from time import time
from scipy import linalg
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2

def _range_normalizer(x, limits=(-1., 1.), margin=1.0):
    x = tf.reshape(x, (-1,))
    min_x = tf.reduce_min(x)
    max_x = tf.reduce_max(x)
    a = margin * ((limits[1] - limits[0]) / (max_x - min_x))
    b = margin * (-(limits[1] - limits[0]) * min_x / (max_x - min_x) + limits[0])
    nr_x = tf.reshape(a * x + b, (128, 128, 1))
    return tf.clip_by_value(nr_x, limits[0], limits[1])

BATCH_SIZE = 16
IMAGE_SIZE = 299



# inception_model = InceptionV3(include_top=False,
#                               weights="imagenet",
#                               pooling="avg",
#                               input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

model = tf.keras.models.load_model("no_aug_imbalanced.h5")


inputs = Input(shape=(128, 128, 1))
x = model.layers[0](inputs)
for layer in model.layers[1:-6]:
    x = layer(x)

out = GlobalAveragePooling2D()(x)

inception_model = Model(inputs=inputs, outputs=out)
# print(inception_model.summary())

print(inception_model.predict(tf.random.normal([1, 128,128, 1])).shape)
print(inception_model.summary())

print(model.layers[0], inception_model.layers[1])

print(model.layers[0].weights == inception_model.layers[1].weights)


def compute_embeddings(dataloader, count):
    image_embeddings = []

    for _ in tqdm(range(count)):
        images = next(iter(dataloader))
        embeddings = inception_model.predict(images)

        image_embeddings.extend(embeddings)

    return np.array(image_embeddings)


def calculate_fid(real_embeddings, generated_embeddings):
    # calculate mean and covariance statistics
    mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
    mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings, rowvar=False)
    # calculate sum squared difference between means

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = linalg.sqrtm(sigma1.dot(sigma2))  # tf.linalg.sqrtm returns nan values for some reason
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


if __name__ == "__main__":
    genloader = dataloader.testloader
    genloader = genloader.map(lambda x, y: _range_normalizer(x, limits=[0., 1.]), num_parallel_calls=tf.data.AUTOTUNE)

    # genloader = genloader.map(lambda x, y: tf.image.grayscale_to_rgb(x),
    #                           num_parallel_calls=tf.data.AUTOTUNE)
    # genloader = genloader.map(
    #     lambda x: tf.image.resize_with_pad(x, target_height=IMAGE_SIZE, target_width=IMAGE_SIZE),
    #     num_parallel_calls=tf.data.AUTOTUNE)
    # genloader = genloader.apply(tf.data.experimental.assert_cardinality(47412))
    genloader = genloader.shuffle(300).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    trainloader = dataloader.trainloader
    trainloader = trainloader.map(lambda x, y: _range_normalizer(x, limits=[0., 1.]), num_parallel_calls=tf.data.AUTOTUNE)

    # trainloader = trainloader.map(lambda x, y: tf.image.grayscale_to_rgb(x),
    #                               num_parallel_calls=tf.data.AUTOTUNE)

    # trainloader = trainloader.map(
    #     lambda x: tf.image.resize_with_pad(x, target_height=IMAGE_SIZE, target_width=IMAGE_SIZE),
    #     num_parallel_calls=tf.data.AUTOTUNE)

    # genloader = genloader.apply(tf.data.experimental.assert_cardinality(47412))
    trainloader = trainloader.shuffle(300).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    count = math.ceil(2300 / BATCH_SIZE)

    # compute embeddings for real images
    real_image_embeddings = compute_embeddings(trainloader, count)

    # compute embeddings for generated images
    generated_image_embeddings = compute_embeddings(genloader, count)

    start = time()
    fid = calculate_fid(real_image_embeddings, generated_image_embeddings)
    print(fid)
    print("time taken for execution: {:.3f}s".format(time() - start))
