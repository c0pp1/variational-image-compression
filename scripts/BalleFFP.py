import tensorflow as tf
import tensorflow_compression as tfc


def get_batched_emodel(batch_shape=()):
    return tfc.ContinuousBatchedEntropyModel(
        prior=tfc.distributions.NoisyDeepFactorized(batch_shape=batch_shape),
        coding_rank=1
    )



class Encoder(tf.keras.layers.Layer):
    """Encoder network for the VAE."""
    
    def __init__(self, N, M, k, format='channel_last'):
        """Initializes the encoder."""
        
        super(Encoder, self).__init__()
        self.N      = N
        self.M      = M
        self.conv1  = tf.keras.layers.Conv2D(self.N, k, strides=2, padding='same', data_format=format)
        self.conv2  = tf.keras.layers.Conv2D(self.N, k, strides=2, data_format=format)
        self.conv3  = tf.keras.layers.Conv2D(self.N, k, strides=1, data_format=format)
        self.conv4  = tf.keras.layers.Conv2D(self.M, k, strides=1, data_format=format)
        self.gdn1   = tfc.layers.GDN()
        self.gdn2   = tfc.layers.GDN()
        self.gdn3   = tfc.layers.GDN()
    
    def call(self, inputs):
        """Forward pass of the encoder."""
        x = self.conv1(inputs)
        x = self.gdn1(x)
        x = self.conv2(x)
        x = self.gdn2(x)
        x = self.conv3(x)
        x = self.gdn3(x)
        z = self.conv4(x)
        del x
        return z
    
    
class Decoder(tf.keras.layers.Layer):
    """Encoder network for the VAE."""
    
    def __init__(self, N, k, c, format):
        """Initializes the encoder."""
        
        super(Decoder, self).__init__()
        self.N      = N
        self.conv2  = tf.keras.layers.Conv2DTranspose(self.N, k, strides=1, data_format=format)
        self.conv1  = tf.keras.layers.Conv2DTranspose(self.N, k, strides=1, data_format=format)
        self.conv3  = tf.keras.layers.Conv2DTranspose(self.N, k, strides=2, data_format=format, output_padding=(1, 1))
        self.conv4  = tf.keras.layers.Conv2DTranspose(c, k, strides=2, data_format=format, padding='same')
        self.gdn1   = tfc.layers.GDN(inverse=True)
        self.gdn2   = tfc.layers.GDN(inverse=True)
        self.gdn3   = tfc.layers.GDN(inverse=True)
    
    def call(self, inputs):
        """Forward pass of the decoder."""
        x = self.conv1(inputs)
        x = self.gdn1(x)
        x = self.conv2(x)
        x = self.gdn2(x)
        x = self.conv3(x)
        x = self.gdn3(x)
        z = self.conv4(x)
        return z
    

class BalleFFP(tf.keras.Model):
    """Encoder network for the VAE."""
    
    def __init__(self, N, M, k2, c, format):
        """Initializes the encoder."""
        
        super(BalleFFP, self).__init__()

        self.bemodel = get_batched_emodel(())
        self.encoder = Encoder(N, M, k2, format)
        self.decoder = Decoder(N, k2, c, format)


    def call(self, inputs, training=True):

        y = self.encoder(inputs)
        y_tilde, rate_b = self.bemodel(y, training=training)

        x_tilde = self.decoder(y_tilde)

        return x_tilde, rate_b