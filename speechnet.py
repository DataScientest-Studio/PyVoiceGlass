import tensorflow as tf
import numpy as np

# ######################################################################################################################
# SpeechNeT Model (advanced model)
# ######################################################################################################################
class AtrousConv1D(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 dilation_rate,
                 use_bias=True,
                 kernel_initializer=tf.keras.initializers.GlorotNormal(),
                 causal=True
                ):
        super(AtrousConv1D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.causal = causal
        # Convolution with dilation
        self.conv1d = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='valid' if causal else 'same',
            use_bias=use_bias,
            kernel_initializer=kernel_initializer
        )
        
    def call(self, inputs):
        # If padding 'valid', the shape of tensor change.
        if self.causal:
            padding = (self.kernel_size - 1) * self.dilation_rate
            inputs = tf.pad(inputs, tf.constant([(0, 0,), (1, 0), (0, 0)]) * padding)
        return self.conv1d(inputs)
    
    
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate, causal, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.batch_normalization = tf.keras.layers.BatchNormalization()
        # First convolution of ResidualBloack
        self.dilated_conv1 = AtrousConv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            causal=causal
        )
        # Second convolution of ResidualBloack
        self.dilated_conv2 = AtrousConv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            causal=causal
        )
        self.out = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=1
        )
        
    def call(self, inputs, training=True):
        # Normalization of data
        data = self.batch_normalization(
            inputs
        )
        # Dilated convolution filters
        filters = self.dilated_conv1(data)
        filters = tf.nn.tanh(filters)
        # Dilated convolution gates
        gates = self.dilated_conv2(data) 
        gates = tf.nn.sigmoid(gates)
        # Elem-wise multiply
        out = tf.nn.tanh(
            self.out(
                filters * gates
            )
        )
        return out + inputs, out
    
        
class ResidualStack(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rates, causal, **kwargs):
        super(ResidualStack, self).__init__(**kwargs)
        # Definition of all Residual Block
        self.blocks = [
            ResidualBlock(
                filters=filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                causal=causal
            )
            for dilation_rate in dilation_rates
        ]
        
    def call(self, inputs, training=True):
        data = inputs
        skip = 0
        for block in self.blocks:
            # Output of Residual Block
            data, current_skip = block(data, training=training)
            # add all each skip connection
            skip += current_skip
        return skip

class SpeechNet(tf.keras.Model):
    def __init__(self, params, **kwargs):
        super(SpeechNet, self).__init__(**kwargs)
        self.batchnormalization1 =tf.keras.layers.BatchNormalization()
        # Expand convolution: extract features
        self.expand = tf.keras.layers.Conv1D(
            filters = params['stack_filters'],
            kernel_size=1,
            padding='same'
        )
        # Definition of all Residual Stack
        self.stacks = [
            ResidualStack(
                filters=params['stack_filters'],
                kernel_size=params['stack_kernel_size'],
                dilation_rates=params['stack_dilation_rates'],
                causal=params['causal_convolutions']
            )
            for _ in range(params['stacks'])
        ]
        # Definition of the last convolution
        self.out = tf.keras.layers.Conv1D(
            filters=len(params['alphabet']) + 1,
            kernel_size=1,
            padding='same'
        )
        self.batchnormalization2 = tf.keras.layers.BatchNormalization()
        
    def call(self, inputs, training=True):
        # Data Normalization
        data = self.batchnormalization1(
            inputs
        )
        # Right shape for convolution.
        if len(data.shape) == 2:
            data = tf.expand_dims(data, 0)
        # Extract features    
        data = self.expand(data)
        # Residual Stack
        for stack in self.stacks:
            data = stack(data, training=training)
        # Data Normalization
        data = self.batchnormalization2(
            data
        )
        return self.out(data) + 1e-8
    
    
def decode_codes(codes, charList):
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            np.arange(len(charList)),
            charList,
            key_dtype=tf.int32
        ),
        '',
        name='id2char'
    )
    return table.lookup(codes)


def greedy_decoder(logits, params):
    # ctc beam search decoder
    predicted_codes, _ = tf.nn.ctc_beam_search_decoder(
        inputs = tf.transpose(logits, (1, 0, 2)),
        sequence_length = [logits.shape[1]]*logits.shape[0],
        beam_width = 100,
        top_paths = 1
    )
    # convert to int32
    codes = tf.cast(predicted_codes[0], tf.int32)
    # Decode the index of caracter
    text = decode_codes(codes, list(params['alphabet']))
    # Convert a SparseTensor to string
    text = tf.sparse.to_dense(text).numpy().astype(str)
    return list(map(lambda x: ''.join(x), text))    