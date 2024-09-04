import os
gpu_num = 0 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Import Sionna
try:
    import sionna
except ImportError as e:
    # Install Sionna if package is not already installed
    import os
    os.system("pip install sionna")
    import sionna

# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')
#matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pickle

from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Conv2D, LayerNormalization
from tensorflow.nn import relu

from sionna.channel.tr38901 import Antenna, AntennaArray, CDL, UMi
from sionna.channel import gen_single_sector_topology as gen_topology
from sionna.channel import OFDMChannel, ApplyOFDMChannel,subcarrier_frequencies
from sionna.mimo import StreamManagement
from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer, RemoveNulledSubcarriers, ResourceGridDemapper
from sionna.utils import BinarySource, ebnodb2no, insert_dims, flatten_last_dims, log10, expand_to_rank
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.mapping import Mapper, Demapper
from ResidualBlock import ResidualBlock


num_conv_channels = 128 #
num_bits_per_symbol = 2 # QPSK


## Target model and receiver definition 

class NeuralReceiver2(Model):
    r"""
    Keras layer implementing a residual convolutional neural receiver.

    This neural receiver is fed with the post-DFT received samples, forming a resource grid of size num_of_symbols x fft_size, and computes LLRs on the transmitted coded bits.
    These LLRs can then be fed to an outer decoder to reconstruct the information bits.

    As the neural receiver is fed with the entire resource grid, including the guard bands and pilots, it also computes LLRs for these resource elements.
    They must be discarded to only keep the LLRs corresponding to the data-carrying resource elements.

    Input
    ------
    y : [batch size, num rx antenna, num ofdm symbols, num subcarriers], tf.complex
        Received post-DFT samples.

    no : [batch size], tf.float32
        Noise variance. At training, a different noise variance value is sampled for each batch example.

    Output
    -------
    : [batch size, num ofdm symbols, num subcarriers, num_bits_per_symbol]
        LLRs on the transmitted bits.
        LLRs computed for resource elements not carrying data (pilots, guard bands...) must be discarded.
    """

def build(self, input_shape):
    
    # Input convolution
    self._input_conv = Conv2D(filters=num_conv_channels,
                            kernel_size=[3,3],
                            padding='same',
                            activation=None)
    # Residual blocks
    self._res_block_1 = ResidualBlock()
    self._res_block_2 = ResidualBlock()
    self._res_block_3 = ResidualBlock()
    self._res_block_4 = ResidualBlock()
    self._res_block_5 = ResidualBlock()
    # Output conv
    self._output_conv = Conv2D(filters=num_bits_per_symbol,
                            kernel_size=[3,3],
                            padding='same',
                            activation=None)
    
def call(self, inputs):
    y, no = inputs
    
    # Feeding the noise power in log10 scale helps with the performance
    no = log10(no)
    
    # Stacking the real and imaginary components of the different antennas along the 'channel' dimension
    y = tf.transpose(y, [0, 2, 3, 1]) # Putting antenna dimension last
    no = insert_dims(no, 3, 1)
    no = tf.tile(no, [1, y.shape[1], y.shape[2], 1])
    # z : [batch size, num ofdm symbols, num subcarriers, 2*num rx antenna + 1]
    z = tf.concat([tf.math.real(y),
                tf.math.imag(y),
                no], axis=-1)
    # Input conv
    z = self._input_conv(z)
    # Residual blocks
    z = self._res_block_1(z)
    z = self._res_block_2(z)
    z = self._res_block_3(z)
    z = self._res_block_4(z)
    z = self._res_block_5(z)
    # Output conv
    z = self._output_conv(z)
    
    return z
    
    
    
