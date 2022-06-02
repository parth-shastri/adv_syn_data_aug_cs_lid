"""

TRAINING SCRIPT - Train the Generator and the discriminator using the WGAN-GP loss.

"""


import tensorflow as tf
import os
from time import time
import datetime
from models.melspecgan import generator_loss, recon_loss, gradient_penalty, discriminator_loss, MSGANGenerator, MSGANCritic
from models.melspecgan import generator_optimizer, discriminator_optimizer
import matplotlib.pyplot as plt
import numpy as np
# tf.config.experimental_run_functions_eagerly(True)
SAMPLE_RATE = 16000
batch_size = 16
d_steps = 5
LAMBDA = 10

# TODO : recode the train step in accordance to the conditional GAN scenario

# model init

"""
Load the dataset from the generated tf.record files.
"""

data_dir = "../input/hindienglish-codeswitched-utterances-tfrecords/Hindi-English_train_utter-tfrec-4sec"


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


filter_nan = lambda x, y: not tf.reduce_any(tf.math.is_nan(y)) and not tf.reduce_any(tf.math.is_nan(x))
filenames = tf.io.gfile.glob(os.path.join(data_dir, "*.tfrec"))
dataset = tf.data.TFRecordDataset(filenames,
                                  num_parallel_reads=tf.data.AUTOTUNE)
dataset = dataset.map(read_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.filter(filter_nan)
# dataset = dataset.apply(tf.data.experimental.assert_cardinality(47412))


dataset = dataset.shuffle(300).batch(batch_size).prefetch(tf.data.AUTOTUNE)

count = 0
for data in dataset:
    count += 1

print("Found {} batches of data...".format(count))

# for (x, y) in dataset.take(1).as_numpy_iterator():
#     print(tf.shape(x[0]), tf.shape(y[0]))
#     print(f"Data stats - \n min value: {np.min(x[0]), np.min(y[0])} \n max value: {np.max(x), np.max(y)}")
#     fig, axs = plt.subplots(nrows=2, constrained_layout=True)
#     X = np.linspace(0, x[0].shape[1], num=x[0].shape[1])
#     Y = range(x[0].shape[0])
#     img = axs[0].pcolormesh(X, Y, np.squeeze(x[0], axis=-1), shading="auto")
#     axs[0].set_title("Mel-spectrogram")
#     axs[0].set_xlabel("time-frames")
#     axs[0].set_ylabel("Mel-frequency scale")
#     fig.colorbar(img, ax=axs[0])
#     axs[1].plot(y[0])
#     axs[1].set_xlim([0, y[0].shape[0]])
#     axs[1].set_title("f0 contour")
#     axs[1].set_xlabel("time-frames")
#     axs[1].set_ylabel("Normalized frequency")
#     fig.suptitle("Data read from the TF-Records format", fontsize=16)
#     fig.show()



#%%

init = tf.keras.initializers.random_normal()

g_model = MSGANGenerator(input_dim=128,
                         model_dim=64,
                         kernel_size=5,
                         dropout=True,
                         kernel_initializer=init,
                         batchnorm=True
                         )
d_model = MSGANCritic(model_dim=64,
                      kernel_size=5,
                      strides=2,
                      kernel_initializer=init,
                      normalization="layer"
                      )

g_optimizer = generator_optimizer
d_optimizer = discriminator_optimizer

ckpt_dir = r"./checkpoints/mel-specgan"
log_dir = r"./logs/mel-specgan"


checkpoint = tf.train.Checkpoint(generator=g_model,
                                 discriminator=d_model,
                                 g_optimizer=g_optimizer,
                                 d_optimizer=d_optimizer)
ckpt_manager = tf.train.CheckpointManager(checkpoint,
                                          directory=ckpt_dir,
                                          max_to_keep=5,
                                          checkpoint_name="melSpecGAN")

prefix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = tf.summary.create_file_writer(
    os.path.join(log_dir, prefix)
)


""" training  """


def get_train_step():
    @tf.function
    def tr_step(g_model,
                d_model,
                g_optimizer,
                d_optimizer,
                real_data,
                gp_weight=10.0,
                d_steps=5):

        image_batch, feature_batch = real_data
        image_batch = tf.clip_by_value(image_batch, -1.0, 1.0)
        batch_size = tf.shape(image_batch)[0]

        for _ in range(d_steps):
            # latent_vector = tf.random.uniform(shape=(batch_size, 128), minval=-1., maxval=1.)

            with tf.GradientTape() as d_tape:
                fake_images = g_model(feature_batch, training=True)
                fake_logits = d_model((fake_images, feature_batch), training=True)

                real_logits = d_model((image_batch, feature_batch), training=True)

                d_cost = discriminator_loss(real_logits, fake_logits)
                gp = gradient_penalty(d_model, image_batch, fake_images, feature_batch, batch_size)

                d_loss = d_cost + gp * gp_weight

            d_gradients = d_tape.gradient(d_loss, d_model.trainable_variables)
            d_optimizer.apply_gradients(zip(d_gradients, d_model.trainable_variables))

        # latent_vector = tf.random.uniform(shape=(batch_size, 128), minval=-1., maxval=1.)

        with tf.GradientTape() as g_tape:
            fake_images = g_model(feature_batch, training=True)

            fake_logits = d_model((fake_images, feature_batch), training=True)

            g_loss = generator_loss(fake_logits) + LAMBDA * recon_loss(fake_images, image_batch)

        g_gradients = g_tape.gradient(g_loss, g_model.trainable_variables)
        g_optimizer.apply_gradients(zip(g_gradients, g_model.trainable_variables))

        return {"g_loss": g_loss, "d_loss": d_loss}

    return tr_step


bn_train_step = get_train_step()
normal_train_step = get_train_step()


def train(generator_model,
          discriminator_model,
          generator_optimizer,
          discriminator_optimizer,
          dataset,
          epochs,
          checkpoint_callback=None,
          summary_writer_callback=None,
          n_critic=5
          ):
    if checkpoint_callback is not None:
        path = checkpoint_callback.restore_or_initialize()
        if path is not None:
            print("Checkpoint restored ! ")
        elif os.path.isdir("../input/cs-melspecgan-run4-checkpoints"):
            name = tf.train.latest_checkpoint("../input/cs-melspecgan-run4-checkpoints/checkpoints/mel-specgan")
            status = checkpoint.restore(name)
            if name is not None and status:
                print("CHECKPOINT RESTORED!")

    global_step = 1  # init the global step

    for epoch in range(epochs):
        start_time = time()
        print(f"Epoch {epoch + 1}/{epochs} [ ", end="")

        gen_loss = tf.keras.metrics.Mean(name="generator_loss")
        disc_loss = tf.keras.metrics.Mean(name="critic_loss")

        for step, (data_batch) in enumerate(dataset):
            loss_dict = normal_train_step(generator_model,
                                          discriminator_model,
                                          generator_optimizer,
                                          discriminator_optimizer,
                                          data_batch,
                                          gp_weight=10.0,
                                          d_steps=n_critic)

            gen_loss.update_state(loss_dict["g_loss"])
            disc_loss.update_state(loss_dict["d_loss"])

            with summary_writer_callback.as_default():
                tf.summary.scalar("generator loss", loss_dict["g_loss"], step=global_step)
                tf.summary.scalar("critic loss", loss_dict["d_loss"], step=global_step)
                if global_step % 10 == 0:
                    fake_spectrogram = generator_model(tf.expand_dims(data_batch[1][0], axis=0))

                    tf.summary.image("generated spectrogram",
                                     tf.image.flip_up_down(fake_spectrogram),
                                     step=global_step)

            if (step + 1) % 1 == 0:
                # generate and save an audio-clip on tensorboard.
                print("\b", end="")
                print("=>", end="")

            global_step += 1  # update the global step

        if checkpoint_callback is not None:
            if (epoch + 1) % 1 == 0:
                checkpoint_callback.save()

        print("\b", end="")  # \b deletes the previous character printed on the console
        print("] - ", end=" ")
        print("{:.1f}s - ".format(time() - start_time), end="")
        print("generator loss : {:.3f}".format(gen_loss.result().numpy()),
              "discriminator loss : {:.3f}".format(disc_loss.result().numpy()))



if __name__ == "__main__":
    train(g_model, d_model, g_optimizer, d_optimizer, dataset, epochs=0,
          summary_writer_callback=summary_writer,
          checkpoint_callback=ckpt_manager,
          n_critic=d_steps)
