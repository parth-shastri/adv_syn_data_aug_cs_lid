
"""Mel Spec GAN"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, ReLU, LeakyReLU, LayerNormalization, BatchNormalization
from tensorflow.keras.layers import Layer, Dropout, Dense, Input, UpSampling2D, Reshape, Concatenate
from tensorflow.keras import Model
from tensorflow_addons.layers import SpectralNormalization

# TODO IMP- update the kaggle code to reflect the modified changes!
# TODO: modify the discriminator normalization scheme. Add spectralNorm or LayerNorm

loss_type = "wgan-gp"
# %%

"""MS Generator"""


class UpsampleConv(Layer):
    def __init__(self, filters, kernel_size, strides=2, upsample=None, batchnorm=None, name=None, **kwargs):
        super(UpsampleConv, self).__init__(name=name)
        self.upsample = upsample
        if batchnorm is not None:
            self.batchnorm = BatchNormalization()
        else:
            self.batchnorm = lambda x, training=None: x

        if self.upsample is not None:
            self.conv = Conv2DTranspose(filters, kernel_size, strides=1, **kwargs)
            if self.upsample == "nn":
                self.up_layer = UpSampling2D(size=strides, interpolation="nearest")
            else:
                raise NotImplementedError(f"The method {self.upsample} is not implemented")
        else:

            self.conv = Conv2DTranspose(filters, kernel_size, strides=strides, **kwargs)

    def call(self, inputs, *args, **kwargs):
        if self.upsample is not None:
            x = self.up_layer(inputs)
            x = self.conv(x)
            out = self.batchnorm(x, **kwargs)

        else:
            x = self.conv(inputs)
            out = self.batchnorm(x, **kwargs)
        return out


class MSGANGenerator(Model):
    def __init__(self,
                 input_dim,
                 output_dim=128,
                 model_dim=64,
                 kernel_size=5,
                 n_channels=1,
                 upsample=None,
                 dropout=False,
                 batchnorm=False,
                 kernel_initializer="glorot_uniform"
                 ):
        super(MSGANGenerator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dim = model_dim
        self.kernel_size = kernel_size
        self.n_channels = n_channels
        self.factors = [1 / 2, 1 / 4, 1 / 8, 1 / 16]
        self.dim_mul = 16
        self.upsample = upsample
        self.dropout = dropout

        self.dense1 = Dense(256 * self.dim, kernel_initializer=kernel_initializer, name="g_dense_1")
        if batchnorm:
            self.batchnorm = BatchNormalization()
        else:
            self.batchnorm = lambda x, training=None: x
        self.conv_layers = []
        for i, factor in enumerate(self.factors):
            self.conv_layers.append(UpsampleConv(self.dim_mul * self.dim * factor,
                                                 kernel_size=self.kernel_size,
                                                 strides=2,
                                                 upsample=self.upsample,
                                                 padding="same",
                                                 kernel_initializer=kernel_initializer,
                                                 name=f"g_upsample_conv_{i}",
                                                 batchnorm=batchnorm
                                                 ))

        self.last_conv = UpsampleConv(self.n_channels,
                                      kernel_size=self.kernel_size,
                                      strides=2,
                                      upsample=self.upsample,
                                      padding="same",
                                      activation="tanh",
                                      kernel_initializer=kernel_initializer,
                                      name="g_last_conv")

    def call(self, inputs, training=None, mask=None):
        x = self.dense1(inputs)
        x = Reshape(target_shape=(4, 4, -1), name="reshape")(x)
        x = self.batchnorm(x, training=training)
        if self.dropout:
            x = Dropout(0.5, name="noise_dropout_0")(x, training=True)
        x = ReLU(name="g_relu_0")(x)
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, training=training)
            if i < 1 and self.dropout:
                x = Dropout(0.5, name=f"noise_dropout_{i + 1}")(x, training=True)
            x = ReLU(name=f"g_relu_{i + 1}")(x)
        out = self.last_conv(x)
        return out

    def model(self):
        inputs = Input((self.input_dim,))
        output = self.call(inputs)
        model = Model(inputs=inputs, outputs=output)
        return model

    def get_config(self):
        config = {"input_dim": self.input_dim, "model_dim": self.dim,
                  "n_channels": self.n_channels, "output_dim": self.output_dim,
                  "kernel_size": self.kernel_size, "upsample": self.upsample,
                  "dropout": self.dropout}
        return config


# %%

"""Discriminator"""


# TODO: make the critic similar to the Cond-GAN setting as in pix2pix i.e. give the conditioning vector to the critic
#  too. NO BATCHNORM IN THE DISCRIMINATOR.

class DownsampleConv(Layer):
    def __init__(self, filters, kernel_size, strides=2, normalization=None, name=None, **kwargs):
        super(DownsampleConv, self).__init__(name=name)
        self.normalization = normalization
        if self.normalization is not None:
            if self.normalization == "layer":
                self.norm = LayerNormalization()
            elif self.normalization == "batch":
                self.norm = BatchNormalization()
            else:
                raise NotImplementedError("Not implemented!")
        else:
            self.norm = lambda x, training=None: x

        self.conv = Conv2D(filters, kernel_size, strides=strides, **kwargs)

    def call(self, inputs, *args, **kwargs):
        x = self.conv(inputs)
        out = self.norm(x, **kwargs)
        return out


class MSGANCritic(Model):
    def __init__(self, model_dim=64,
                 kernel_size=5,
                 strides=2,
                 kernel_initializer="glorot_uniform",
                 normalization=None
                 ):
        super(MSGANCritic, self).__init__()
        self.dim = model_dim
        self.kernel_size = kernel_size
        self.strides = strides
        self.factors = [1, 2, 4, 8, 16]
        self.conv_layers = []

        self.dense_pre = Dense(128 * 128 * 1, kernel_initializer=kernel_initializer, name="prenet_dense_1")
        self.dense_out = Dense(1, kernel_initializer=kernel_initializer, name="out")

        for i, factor in enumerate(self.factors):
            self.conv_layers.append(DownsampleConv(factor * self.dim, self.kernel_size,
                                                   strides=self.strides,
                                                   padding="same",
                                                   kernel_initializer=kernel_initializer,
                                                   name=f"d_conv_{i}",
                                                   normalization=normalization))

    def call(self, inputs, training=None, mask=None):
        spec_input, f0 = inputs
        # pre-net
        x = self.dense_pre(f0)
        x = Reshape(target_shape=(128, 128, 1), name="prenet_reshape")(x)
        x = Concatenate(name="concat")([spec_input, x])

        for i, conv in enumerate(self.conv_layers):
            x = conv(x, training=training)

            x = LeakyReLU(alpha=0.2, name=f"lrelu_{i}")(x)

        x = Reshape(target_shape=(256 * self.dim,), name="reshape")(x)
        out = self.dense_out(x)
        return out

    def model(self):
        input1 = Input(shape=(128, 128, 1))
        input2 = Input(shape=(128,))
        out = self.call([input1, input2])
        model = Model(inputs=[input1, input2], outputs=out)
        return model

    def get_config(self):
        config = {"model_dim": self.dim,
                  "kernel_size": self.kernel_size,
                  "strides": self.strides,
                  }
        return config


# %%

"""Loss and Optimizers"""


def generator_loss_nsgan(fake_logits):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(fake_logits),
                                                                  fake_logits))
    return loss


def discriminator_loss_nsgan(real_logits, fake_logits):
    D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(real_logits),
                                                                    real_logits))
    D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(fake_logits),
                                                                     fake_logits))
    return D_loss / 2.


def critic_loss(real_logits, fake_logits):
    real_loss = tf.reduce_mean(real_logits)
    fake_loss = tf.reduce_mean(fake_logits)
    return fake_loss - real_loss


def generator_loss_wgan(fake_logits):
    fake_loss = tf.reduce_mean(fake_logits)
    return -fake_loss


def gradient_penalty(critic_net, real_images, fake_images, feature_batch, batch_size):
    alpha = tf.random.uniform((batch_size, 1, 1, 1), 0.0, 1.0)

    diff = fake_images - real_images
    interpolated = real_images + (diff * alpha)

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = critic_net((interpolated, feature_batch), training=True)

    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
    gp = tf.reduce_mean((norm - 1) ** 2)
    return gp


def recon_loss(gen_image, real_image):
    l1_loss = tf.reduce_mean(tf.abs(real_image - gen_image))
    return l1_loss


if loss_type == "wgan-gp":
    def generator_loss(x):
        return generator_loss_wgan(x)


    def discriminator_loss(x, y):
        return critic_loss(x, y)


    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5,
                                                   beta_2=0.9)  # HP from the SpecGAN paper
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5,
                                                       beta_2=0.9)  # HP from the SpecGAN paper

elif loss_type == "nsgan":
    def generator_loss(x):
        return generator_loss_nsgan(x)


    def discriminator_loss(x, y):
        return discriminator_loss_nsgan(x, y)


    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4,
                                                   beta_1=0.5)  # HP from the DCGAN paper
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4,
                                                       beta_1=0.5)

if __name__ == "__main__":
    # test code
    vector = tf.random.uniform((1, 128))
    gen = MSGANGenerator(128,
                         dropout=True,
                         batchnorm=True)  # used dropout as in the pix-2-pix paper
    
    print(gen.model().summary())
    
    gen_out = gen(vector, training=True)

    # test_code
    inp = (tf.random.normal((1, 128, 128, 1)), tf.random.normal((1, 128)))
    try:
        critic = MSGANCritic(model_dim=64,
                             normalization="layer")
        print(critic.model().summary())
        critic_out = critic(inp)

    except NotImplementedError:
        print("detected")
# %% [code]
"""Mel Spec GAN w/ PatchGAN Discriminator"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, ReLU, LeakyReLU, LayerNormalization, BatchNormalization
from tensorflow.keras.layers import Layer, Dropout, Dense, Input, UpSampling2D, Reshape, Concatenate
from tensorflow.keras import Model

# TODO IMP- update the kaggle code to reflect the modified changes!
# TODO: modify the discriminator normalization scheme. Add spectralNorm or LayerNorm

loss_type = "nsgan"
# %%

"""MS Generator"""


class UpsampleConv(Layer):
    def __init__(self, filters, kernel_size, strides=2, upsample=None, batchnorm=None, name=None, **kwargs):
        super(UpsampleConv, self).__init__(name=name)
        self.upsample = upsample
        if batchnorm is not None:
            self.batchnorm = BatchNormalization()
        else:
            self.batchnorm = lambda x, training=None: x

        if self.upsample is not None:
            self.conv = Conv2DTranspose(filters, kernel_size, strides=1, **kwargs)
            if self.upsample == "nn":
                self.up_layer = UpSampling2D(size=strides, interpolation="nearest")
            else:
                raise NotImplementedError(f"The method {self.upsample} is not implemented")
        else:

            self.conv = Conv2DTranspose(filters, kernel_size, strides=strides, **kwargs)

    def call(self, inputs, *args, **kwargs):
        if self.upsample is not None:
            x = self.up_layer(inputs)
            x = self.conv(x)
            out = self.batchnorm(x, **kwargs)

        else:
            x = self.conv(inputs)
            out = self.batchnorm(x, **kwargs)
        return out


class MSGANGenerator(Model):
    def __init__(self,
                 input_dim,
                 output_dim=128,
                 model_dim=64,
                 kernel_size=5,
                 n_channels=1,
                 upsample=None,
                 dropout=False,
                 batchnorm=False,
                 kernel_initializer="glorot_uniform"
                 ):
        super(MSGANGenerator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dim = model_dim
        self.kernel_size = kernel_size
        self.n_channels = n_channels
        self.factors = [1 / 2, 1 / 4, 1 / 8, 1 / 16]
        self.dim_mul = 16
        self.upsample = upsample
        self.dropout = dropout

        self.dense1 = Dense(256 * self.dim, kernel_initializer=kernel_initializer, name="g_dense_1")
        if batchnorm:
            self.batchnorm = BatchNormalization()
        else:
            self.batchnorm = lambda x, training=None: x
        self.conv_layers = []
        for i, factor in enumerate(self.factors):
            self.conv_layers.append(UpsampleConv(self.dim_mul * self.dim * factor,
                                                 kernel_size=self.kernel_size,
                                                 strides=2,
                                                 upsample=self.upsample,
                                                 padding="same",
                                                 kernel_initializer=kernel_initializer,
                                                 name=f"g_upsample_conv_{i}",
                                                 batchnorm=batchnorm
                                                 ))

        self.last_conv = UpsampleConv(self.n_channels,
                                      kernel_size=self.kernel_size,
                                      strides=2,
                                      upsample=self.upsample,
                                      padding="same",
                                      activation="tanh",
                                      kernel_initializer=kernel_initializer,
                                      name="g_last_conv")

    def call(self, inputs, training=None, mask=None):
        x = self.dense1(inputs)
        x = Reshape(target_shape=(4, 4, -1), name="reshape")(x)
        x = self.batchnorm(x, training=training)
        if self.dropout:
            x = Dropout(0.5, name="noise_dropout_0")(x, training=True)
        x = ReLU(name="g_relu_0")(x)
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, training=training)
            if i < 1 and self.dropout:
                x = Dropout(0.5, name=f"noise_dropout_{i + 1}")(x, training=True)
            x = ReLU(name=f"g_relu_{i + 1}")(x)
        out = self.last_conv(x)
        return out

    def model(self):
        inputs = Input((self.input_dim,))
        output = self.call(inputs)
        model = Model(inputs=inputs, outputs=output)
        return model

    def get_config(self):
        config = {"input_dim": self.input_dim, "model_dim": self.dim,
                  "n_channels": self.n_channels, "output_dim": self.output_dim,
                  "kernel_size": self.kernel_size, "upsample": self.upsample,
                  "dropout": self.dropout}
        return config


# %%

"""Discriminator"""


# TODO: make the critic similar to the Cond-GAN setting as in pix2pix i.e. give the conditioning vector to the critic
# too. NO BATCHNORM IN THE DISCRIMINATOR.

class DownsampleConv(Layer):
    def __init__(self, filters, kernel_size, strides=2, normalization=None, name=None, **kwargs):
        super(DownsampleConv, self).__init__(name=name)
        self.normalization = normalization
        if self.normalization is not None:
            if self.normalization == "layer":
                self.norm = LayerNormalization()
            elif self.normalization == "batch":
                self.norm = BatchNormalization()
            else:
                raise NotImplementedError("Not implemented!")
        else:
            self.norm = lambda x, training=None: x

        self.conv = Conv2D(filters, kernel_size, strides=strides, **kwargs)

    def call(self, inputs, *args, **kwargs):
        x = self.conv(inputs)
        out = self.norm(x, **kwargs)
        return out


class PatchGANDisc(Model):
    def __init__(self, model_dim=64,
                 kernel_size=5,
                 strides=2,
                 kernel_initializer="glorot_uniform",
                 normalization=None
                 ):
        super(PatchGANDisc, self).__init__()
        self.dim = model_dim
        self.kernel_size = kernel_size
        self.strides = strides
        self.factors = [1, 2, 4]
        self.conv_layers = []

        self.dense_pre = Dense(128 * 128 * 1, kernel_initializer=kernel_initializer, name="prenet_dense_1")
        self.dense_out = Dense(1, kernel_initializer=kernel_initializer, name="out")

        for i, factor in enumerate(self.factors):
            self.conv_layers.append(DownsampleConv(factor * self.dim, self.kernel_size,
                                                   strides=self.strides,
                                                   padding="same",
                                                   kernel_initializer=kernel_initializer,
                                                   name=f"d_conv_{i}",
                                                   normalization=normalization))
        self.penult_conv = Conv2D(self.dim * 8, kernel_size=4, strides=1, name="penult_conv",
                                  kernel_initializer=kernel_initializer)
        self.pad = tf.keras.layers.ZeroPadding2D(padding=(1, 1))
        self.patchgan_conv = Conv2D(1, kernel_size=4, strides=1, name="patchgan_conv",
                                    kernel_initializer=kernel_initializer)

    def call(self, inputs, training=None, mask=None):
        spec_input, f0 = inputs
        # pre-net
        x = self.dense_pre(f0)
        x = Reshape(target_shape=(128, 128, 1), name="prenet_reshape")(x)
        x = Concatenate(name="concat")([spec_input, x])

        for i, conv in enumerate(self.conv_layers):
            x = conv(x, training=training)

            x = LeakyReLU(alpha=0.2, name=f"lrelu_{i}")(x)

        x = self.pad(x)
        x = self.penult_conv(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = self.pad(x)
        out = self.patchgan_conv(x)
        return out

    def model(self):
        input1 = Input(shape=(128, 128, 1))
        input2 = Input(shape=(128,))
        out = self.call([input1, input2])
        model = Model(inputs=[input1, input2], outputs=out)
        return model

    def get_config(self):
        config = {"model_dim": self.dim,
                  "kernel_size": self.kernel_size,
                  "strides": self.strides,
                  }
        return config


# %%

"""Loss and Optimizers"""


def generator_loss_nsgan(fake_logits):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(fake_logits),
                                                                  fake_logits))
    return loss


def discriminator_loss_nsgan(real_logits, fake_logits):
    D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(real_logits),
                                                                    real_logits))
    D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(fake_logits),
                                                                     fake_logits))
    return D_loss / 2.


def critic_loss(real_logits, fake_logits):
    real_loss = tf.reduce_mean(real_logits)
    fake_loss = tf.reduce_mean(fake_logits)
    return fake_loss - real_loss


def generator_loss_wgan(fake_logits):
    fake_loss = tf.reduce_mean(fake_logits)
    return -fake_loss


def gradient_penalty(critic_net, real_images, fake_images, feature_batch, batch_size):
    alpha = tf.random.uniform((batch_size, 1, 1, 1), 0.0, 1.0)

    diff = fake_images - real_images
    interpolated = real_images + (diff * alpha)

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = critic_net((interpolated, feature_batch), training=True)

    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
    gp = tf.reduce_mean((norm - 1) ** 2)
    return gp


def recon_loss(gen_image, real_image):
    l1_loss = tf.reduce_mean(tf.abs(real_image - gen_image))
    return l1_loss


if loss_type == "wgan-gp":
    def generator_loss(x):
        return generator_loss_wgan(x)


    def discriminator_loss(x, y):
        return critic_loss(x, y)


    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5,
                                                   beta_2=0.9)  # HP from the SpecGAN paper
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5,
                                                       beta_2=0.9)  # HP from the SpecGAN paper

elif loss_type == "nsgan":
    def generator_loss(x):
        return generator_loss_nsgan(x)


    def discriminator_loss(x, y):
        return discriminator_loss_nsgan(x, y)


    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4,
                                                   beta_1=0.5)  # HP from the DCGAN paper
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4,
                                                       beta_1=0.5)

if __name__ == "__main__":
    # test code
    vector = tf.random.uniform((1, 128))
    gen = MSGANGenerator(128,
                         dropout=True,
                         batchnorm=True)  # used dropout as in the pix-2-pix paper

    print(gen.model().summary())

    gen_out = gen(vector, training=True)

    # test_code
    inp = (tf.random.normal((1, 128, 128, 1)), tf.random.normal((1, 128)))
    try:
        critic = PatchGANDisc(model_dim=64,
                              normalization="fancy")
        print(critic.model().summary())
        critic_out = critic(inp)

    except NotImplementedError:
        print("detected")

