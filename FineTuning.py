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


####

#matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pickle

from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Conv2D, LayerNormalization
from tensorflow.nn import relu
from tensorflow.keras import Model

from sionna.channel.tr38901 import Antenna, AntennaArray, CDL
from sionna.channel import OFDMChannel
from sionna.mimo import StreamManagement
from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer, RemoveNulledSubcarriers, ResourceGridDemapper
from sionna.utils import BinarySource, ebnodb2no, insert_dims, flatten_last_dims, log10, expand_to_rank
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.mapping import Mapper, Demapper
from sionna.utils.metrics import compute_ber
from sionna.utils import sim_ber
from NeuralReceiver import NeuralReceiver
from NeuralReceiver_2 import NeuralReceiver2

#####
#def fine_tuning(target_model, weight_filepath):
def fine_tuning(chann_model ,subc_space, bit_per_sys, alf,source_scenario, source_filepath):
############################################
    ## Channel configuration
    carrier_frequency = 3.5e9 # Hz
    delay_spread = 100e-9 # s
    cdl_model = chann_model # CDL model to use
    speed = 10.0 # Speed for evaluation and training [m/s]
    # SNR range for evaluation and training [dB]
    ebno_db_min = -5.0
    ebno_db_max = 10.0

    ############################################
    ## OFDM waveform configuration
    subcarrier_spacing = subc_space #120e3# 60e3#30e3 # Hz
    fft_size = 128 # Number of subcarriers forming the resource grid, including the null-subcarrier and the guard bands
    num_ofdm_symbols = 14 # Number of OFDM symbols forming the resource grid
    dc_null = True # Null the DC subcarrier
    num_guard_carriers = [5, 6] # Number of guard carriers on each side
    pilot_pattern = "kronecker" # Pilot pattern
    pilot_ofdm_symbol_indices = [2, 11] # Index of OFDM symbols carrying pilots
    cyclic_prefix_length = 0 # Simulation in frequency domain. This is useless

    ############################################
    ## Modulation and coding configuration
    num_bits_per_symbol = bit_per_sys#2 # QPSK
    coderate = 0.5 # Coderate for LDPC code

    ############################################
    ## Neural receiver configuration
    num_conv_channels = 128 # Number of convolutional channels for the convolutional layers forming the neural receiver

    ############################################
    ## Training configuration
    num_training_iterations = np.int(alf*30000)# Number of training iterations
    training_batch_size = 128 # Training batch size
    model_weights_path = "feature_ext_weights" # Location to save the neural receiver weights once training is done

    ############################################
    ## Evaluation configuration
    #results_filename = "nnrx_results_layer_plus" # Location to save the results


    #####

    stream_manager = StreamManagement(np.array([[1]]), # Receiver-transmitter association matrix
                                    1)               # One stream per transmitter


    ######

    resource_grid = ResourceGrid(num_ofdm_symbols = num_ofdm_symbols,
                                fft_size = fft_size,
                                subcarrier_spacing = subcarrier_spacing,
                                num_tx = 1,
                                num_streams_per_tx = 1,
                                cyclic_prefix_length = cyclic_prefix_length,
                                dc_null = dc_null,
                                pilot_pattern = pilot_pattern,
                                pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices,
                                num_guard_carriers = num_guard_carriers) 

    # Codeword length. It is calculated from the total number of databits carried by the resource grid, and the number of bits transmitted per resource element
    n = int(resource_grid.num_data_symbols*num_bits_per_symbol)
    # Number of information bits per codeword
    k = int(n*coderate) 
    #####

    ut_antenna = Antenna(polarization="single",
                        polarization_type="V",
                        antenna_pattern="38.901",
                        carrier_frequency=carrier_frequency)

    bs_array = AntennaArray(num_rows=1,
                            num_cols=1,
                            polarization="dual",
                            polarization_type="VH",
                            antenna_pattern="38.901",
                            carrier_frequency=carrier_frequency)

    #####


    
    #####

   

    #####

    class E2ESystem(Model):
        r"""
        Keras model that implements the end-to-end systems.
        
        As the three considered end-to-end systems (perfect CSI baseline, LS estimation baseline, and neural receiver) share most of
        the link components (transmitter, channel model, outer code...), they are implemented using the same Keras model.

        When instantiating the Keras model, the parameter ``system`` is used to specify the system to setup,
        and the parameter ``training`` is used to specified if the system is instantiated to be trained or to be evaluated.
        The ``training`` parameter is only relevant when the neural 
        
        At each call of this model:
        * A batch of codewords is randomly sampled, modulated, and mapped to resource grids to form the channel inputs
        * A batch of channel realizations is randomly sampled and applied to the channel inputs
        * The receiver is executed on the post-DFT received samples to compute LLRs on the coded bits.
        Which receiver is executed (baseline with perfect CSI knowledge, baseline with LS estimation, or neural receiver) depends
        on the specified ``system`` parameter.
        * If not training, the outer decoder is applied to reconstruct the information bits
        * If training, the BMD rate is estimated over the batch from the LLRs and the transmitted bits
        
        Parameters
        -----------
        system : str
            Specify the receiver to use. Should be one of 'baseline-perfect-csi', 'baseline-ls-estimation' or 'neural-receiver'
        
        training : bool
            Set to `True` if the system is instantiated to be trained. Set to `False` otherwise. Defaults to `False`.
            If the system is instantiated to be trained, the outer encoder and decoder are not instantiated as they are not required for training.
            This significantly reduces the computational complexity of training.
            If training, the bit-metric decoding (BMD) rate is computed from the transmitted bits and the LLRs. The BMD rate is known to be
            an achievable information rate for BICM systems, and therefore training of the neural receiver aims at maximizing this rate.
        
        Input
        ------
        batch_size : int
            Batch size
        
        no : scalar or [batch_size], tf.float
            Noise variance.
            At training, a different noise variance should be sampled for each batch example.
        
        Output
        -------
        If ``training`` is set to `True`, then the output is a single scalar, which is an estimation of the BMD rate computed over the batch. It
        should be used as objective for training.
        If ``training`` is set to `False`, the transmitted information bits and their reconstruction on the receiver side are returned to
        compute the block/bit error rate. 
        """
        
        def __init__(self, system, training=False):
            super().__init__()
            self._system = system
            self._training = training
        
            ######################################
            ## Transmitter
            self._binary_source = BinarySource()
            # To reduce the computational complexity of training, the outer code is not used when training,
            # as it is not required
            if not training:
                self._encoder = LDPC5GEncoder(k, n)
            self._mapper = Mapper("qam", num_bits_per_symbol)
            self._rg_mapper = ResourceGridMapper(resource_grid)
            
            ######################################
            ## Channel
            # A 3GPP CDL channel model is used
            cdl = CDL(cdl_model, delay_spread, carrier_frequency,
                    ut_antenna, bs_array, "uplink", min_speed=speed)
            self._channel = OFDMChannel(cdl, resource_grid, normalize_channel=True, return_channel=True)
            
            ######################################
            ## Receiver
            # Three options for the receiver depending on the value of `system`
            if "baseline" in system:
                if system == 'baseline-perfect-csi': # Perfect CSI
                    self._removed_null_subc = RemoveNulledSubcarriers(resource_grid)
                elif system == 'baseline-ls-estimation': # LS estimation
                    self._ls_est = LSChannelEstimator(resource_grid, interpolation_type="nn")
                # Components required by both baselines
                self._lmmse_equ = LMMSEEqualizer(resource_grid, stream_manager, )
                self._demapper = Demapper("app", "qam", num_bits_per_symbol)
            elif system == "neural-receiver": # Neural receiver
                self._neural_receiver = NeuralReceiver()
                self._rg_demapper = ResourceGridDemapper(resource_grid, stream_manager) # Used to extract data-carrying resource elements
            # To reduce the computational complexity of training, the outer code is not used when training,
            # as it is not required
            if not training:
                self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)
        
        @tf.function
        def call(self, batch_size, ebno_db):
            
            # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
            if len(ebno_db.shape) == 0:
                ebno_db = tf.fill([batch_size], ebno_db)
            
            ######################################
            ## Transmitter
            no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
            # Outer coding is only performed if not training
            if self._training:
                c = self._binary_source([batch_size, 1, 1, n])
            else:
                b = self._binary_source([batch_size, 1, 1, k])
                c = self._encoder(b)
            # Modulation
            x = self._mapper(c)
            x_rg = self._rg_mapper(x)
            
            ######################################
            ## Channel
            # A batch of new channel realizations is sampled and applied at every inference
            no_ = expand_to_rank(no, tf.rank(x_rg))
            y,h = self._channel([x_rg, no_])
            
            ######################################
            ## Receiver       
            # Three options for the receiver depending on the value of ``system``
            if "baseline" in self._system:
                if self._system == 'baseline-perfect-csi':
                    h_hat = self._removed_null_subc(h) # Extract non-null subcarriers
                    err_var = 0.0 # No channel estimation error when perfect CSI knowledge is assumed
                elif self._system == 'baseline-ls-estimation':
                    h_hat, err_var = self._ls_est([y, no]) # LS channel estimation with nearest-neighbor
                x_hat, no_eff = self._lmmse_equ([y, h_hat, err_var, no]) # LMMSE equalization
                no_eff_= expand_to_rank(no_eff, tf.rank(x_hat))
                llr = self._demapper([x_hat, no_eff_]) # Demapping
            elif self._system == "neural-receiver":
                # The neural receover computes LLRs from the frequency domain received symbols and N0
                y = tf.squeeze(y, axis=1)
                llr = self._neural_receiver([y, no])
                llr = insert_dims(llr, 2, 1) # Reshape the input to fit what the resource grid demapper is expected
                llr = self._rg_demapper(llr) # Extract data-carrying resource elements. The other LLrs are discarded
                llr = tf.reshape(llr, [batch_size, 1, 1, n]) # Reshape the LLRs to fit what the outer decoder is expected

            # Outer coding is not needed if the information rate is returned
            if self._training:
                # Compute and return BMD rate (in bit), which is known to be an achievable
                # information rate for BICM systems.
                # Training aims at maximizing the BMD rate
                bce = tf.nn.sigmoid_cross_entropy_with_logits(c, llr)
                bce = tf.reduce_mean(bce)
                rate = tf.constant(1.0, tf.float32) - bce/tf.math.log(2.)
                return rate
            else:
                # Outer decoding
                b_hat = self._decoder(llr)
                return b,b_hat # Ground truth and reconstructed information bits returned for BER/BLER computation

    #####################################################################################################################################################################
    #######################################################################################################################################################################

    # parameters for evaluation
    ebno_db_min = -5.0
    ebno_db_max = 10.0
    ebno_dbs = np.arange(ebno_db_min, # Min SNR for evaluation
                        ebno_db_max, # Max SNR for evaluation
                        0.5) # Step
    """
    # Evaluate baselines
    BLER_perf_csi={}
    BLER_ls={}

    model = E2ESystem('baseline-perfect-csi')
    _,bler = sim_ber(model, ebno_dbs, batch_size=128, num_target_block_errors=100, max_mc_iter=10000)
    BLER_perf_csi['baseline-perfect-csi'] = bler.numpy()

    model = E2ESystem('baseline-ls-estimation')
    _,bler = sim_ber(model, ebno_dbs, batch_size=128, num_target_block_errors=100, max_mc_iter=10000)
    BLER_ls['baseline-ls-estimation'] = bler.numpy()
    """

    ################################################################################################################################################################### 
    ###################################################################################################################################################################
    # an instance of source model to build layers        

    ## an instance of of the model and loading saved wieghts
    model = E2ESystem('neural-receiver',training = True)
    model(1, tf.constant(10.0, tf.float32))
    #model.load_weights("/home/es.aau.dk/mw88bt/wght_C15kHz",by_name=True, skip_mismatch=True) # wieights of base line or source model CDL C
    model.load_weights(source_filepath,by_name=True, skip_mismatch=True) # wieights of base line or source model CDL C

    ### assigning the nueral receiver layers to a variable manually

    model_nnrx1 = model.layers[4].layers[0]
    model_nnrx2 = model.layers[4].layers[1]
    model_nnrx3 = model.layers[4].layers[2]
    model_nnrx4 = model.layers[4].layers[3]
    model_nnrx5 = model.layers[4].layers[4]

    ########################################################################################################################################################################
    ##########################################################################################################################

    ####

    

    class E2ESystem2(Model):
        r"""
        Keras model that implements the end-to-end systems.
        
        As the three considered end-to-end systems (perfect CSI baseline, LS estimation baseline, and neural receiver) share most of
        the link components (transmitter, channel model, outer code...), they are implemented using the same Keras model.

        When instantiating the Keras model, the parameter ``system`` is used to specify the system to setup,
        and the parameter ``training`` is used to specified if the system is instantiated to be trained or to be evaluated.
        The ``training`` parameter is only relevant when the neural 
        
        At each call of this model:
        * A batch of codewords is randomly sampled, modulated, and mapped to resource grids to form the channel inputs
        * A batch of channel realizations is randomly sampled and applied to the channel inputs
        * The receiver is executed on the post-DFT received samples to compute LLRs on the coded bits.
        Which receiver is executed (baseline with perfect CSI knowledge, baseline with LS estimation, or neural receiver) depends
        on the specified ``system`` parameter.
        * If not training, the outer decoder is applied to reconstruct the information bits
        * If training, the BMD rate is estimated over the batch from the LLRs and the transmitted bits
        
        Parameters
        -----------
        system : str
            Specify the receiver to use. Should be one of 'baseline-perfect-csi', 'baseline-ls-estimation' or 'neural-receiver'
        
        training : bool
            Set to `True` if the system is instantiated to be trained. Set to `False` otherwise. Defaults to `False`.
            If the system is instantiated to be trained, the outer encoder and decoder are not instantiated as they are not required for training.
            This significantly reduces the computational complexity of training.
            If training, the bit-metric decoding (BMD) rate is computed from the transmitted bits and the LLRs. The BMD rate is known to be
            an achievable information rate for BICM systems, and therefore training of the neural receiver aims at maximizing this rate.
        
        Input
        ------
        batch_size : int
            Batch size
        
        no : scalar or [batch_size], tf.float
            Noise variance.
            At training, a different noise variance should be sampled for each batch example.
        
        Output
        -------
        If ``training`` is set to `True`, then the output is a single scalar, which is an estimation of the BMD rate computed over the batch. It
        should be used as objective for training.
        If ``training`` is set to `False`, the transmitted information bits and their reconstruction on the receiver side are returned to
        compute the block/bit error rate. 
        """
        
        def __init__(self, system, training=False):
            super().__init__()
            self._system = system
            self._training = training
        
            ######################################
            ## Transmitter
            self._binary_source = BinarySource()
            # To reduce the computational complexity of training, the outer code is not used when training,
            # as it is not required
            if not training:
                self._encoder = LDPC5GEncoder(k, n)
            self._mapper = Mapper("qam", num_bits_per_symbol)
            self._rg_mapper = ResourceGridMapper(resource_grid)
            
            ######################################
            ## Channel
            # A 3GPP CDL channel model is used
            cdl = CDL(cdl_model, delay_spread, carrier_frequency,
                    ut_antenna, bs_array, "uplink", min_speed=speed)
            self._channel = OFDMChannel(cdl, resource_grid, normalize_channel=True, return_channel=True)
            
            ######################################
            ## Receiver
            # Three options for the receiver depending on the value of `system`
            if "baseline" in system:
                if system == 'baseline-perfect-csi': # Perfect CSI
                    self._removed_null_subc = RemoveNulledSubcarriers(resource_grid)
                elif system == 'baseline-ls-estimation': # LS estimation
                    self._ls_est = LSChannelEstimator(resource_grid, interpolation_type="nn")
                # Components required by both baselines
                self._lmmse_equ = LMMSEEqualizer(resource_grid, stream_manager, )
                self._demapper = Demapper("app", "qam", num_bits_per_symbol)
            elif system == "neural-receiver": # Neural receiver
                self._neural_receiver = NeuralReceiver2()
                self._rg_demapper = ResourceGridDemapper(resource_grid, stream_manager) # Used to extract data-carrying resource elements
            # To reduce the computational complexity of training, the outer code is not used when training,
            # as it is not required
            if not training:
                self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)
        
        @tf.function
        def call(self, batch_size, ebno_db):
            
            # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
            if len(ebno_db.shape) == 0:
                ebno_db = tf.fill([batch_size], ebno_db)
            
            ######################################
            ## Transmitter
            no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
            # Outer coding is only performed if not training
            if self._training:
                c = self._binary_source([batch_size, 1, 1, n])
            else:
                b = self._binary_source([batch_size, 1, 1, k])
                c = self._encoder(b)
            # Modulation
            x = self._mapper(c)
            x_rg = self._rg_mapper(x)
            
            ######################################
            ## Channel
            # A batch of new channel realizations is sampled and applied at every inference
            no_ = expand_to_rank(no, tf.rank(x_rg))
            y,h = self._channel([x_rg, no_])
            
            ######################################
            ## Receiver       
            # Three options for the receiver depending on the value of ``system``
            if "baseline" in self._system:
                if self._system == 'baseline-perfect-csi':
                    h_hat = self._removed_null_subc(h) # Extract non-null subcarriers
                    err_var = 0.0 # No channel estimation error when perfect CSI knowledge is assumed
                elif self._system == 'baseline-ls-estimation':
                    h_hat, err_var = self._ls_est([y, no]) # LS channel estimation with nearest-neighbor
                x_hat, no_eff = self._lmmse_equ([y, h_hat, err_var, no]) # LMMSE equalization
                no_eff_= expand_to_rank(no_eff, tf.rank(x_hat))
                llr = self._demapper([x_hat, no_eff_]) # Demapping
            elif self._system == "neural-receiver":
                # The neural receover computes LLRs from the frequency domain received symbols and N0
                y = tf.squeeze(y, axis=1)
                llr = self._neural_receiver([y, no])
                llr = insert_dims(llr, 2, 1) # Reshape the input to fit what the resource grid demapper is expected
                llr = self._rg_demapper(llr) # Extract data-carrying resource elements. The other LLrs are discarded
                llr = tf.reshape(llr, [batch_size, 1, 1, n]) # Reshape the LLRs to fit what the outer decoder is expected

            # Outer coding is not needed if the information rate is returned
            if self._training:
                # Compute and return BMD rate (in bit), which is known to be an achievable
                # information rate for BICM systems.
                # Training aims at maximizing the BMD rate
                bce = tf.nn.sigmoid_cross_entropy_with_logits(c, llr)
                bce = tf.reduce_mean(bce)
                rate = tf.constant(1.0, tf.float32) - bce/tf.math.log(2.)
                return rate
            else:
                # Outer decoding
                b_hat = self._decoder(llr)
                return b,b_hat # Ground truth and reconstructed information bits returned for BER/BLER computation
                
    ##################################################################################################################################################
    #################################################################################################################################
    #################################################################################################################################
    ##### create an instance of the source model and use it as target model without additional layers

    ## an instance of of the model and loading saved wieghts
    model_tf_all = E2ESystem('neural-receiver',training = True)
    model_tf_all(1, tf.constant(10.0, tf.float32))
    model_tf_all.load_weights(source_filepath,by_name=True, skip_mismatch=True)

    #### training of target model

    optimizer = tf.keras.optimizers.Adam()
    ## training of the target model
    for i in range(num_training_iterations):
        # Sampling a batch of SNRs
        ebno_db = tf.random.uniform(shape=[], minval=ebno_db_min, maxval=ebno_db_max)
        # Forward pass
        with tf.GradientTape() as tape:
            rate = model_tf_all(training_batch_size, ebno_db)
            # Tensorflow optimizers only know how to minimize loss function.
            # Therefore, a loss function is defined as the additive inverse of the BMD rate
            loss = -rate
        # Computing and applying gradients        
        weights = model_tf_all.trainable_weights
        grads = tape.gradient(loss, weights)
        optimizer.apply_gradients(zip(grads, weights))
        # Periodically printing the progress
        if i % 100 == 0:
            print('Iteration {}/{}  Rate: {:.4f} bit'.format(i, num_training_iterations, rate.numpy()), end='\r')

    #Save the weights in a file
    weights = model_tf_all.get_weights()
    with open('finetune_all_wgts.pkl', 'wb') as f:
        pickle.dump(weights, f)
        
    ## save the weights in a hd5 format
    #model_tf_b2a_all.save_weights('weights_tf_b2a_all',save_format='h5')



    # parameters for evaluation
    ebno_db_min = -5.0
    ebno_db_max = 10.0
    ebno_dbs = np.arange(ebno_db_min, # Min SNR for evaluation
                        ebno_db_max, # Max SNR for evaluation
                        0.5) # Step





    ####
    # run an instance of the model 
    model_tf_all= E2ESystem('neural-receiver')

    # Run one inference to build the layers and loading the weights from the target model
    model_tf_all(1, tf.constant(10.0, tf.float32))
    with open('finetune_all_wgts.pkl', 'rb') as f:
        weights = pickle.load(f)
    model_tf_all.set_weights(weights)
    #model_tf_b2a_all.load_weights("/home/es.aau.dk/mw88bt/weights_tf_b2a_all")



    BLER_all={}

    # Evaluations of the target model
    _,bler = sim_ber(model_tf_all, ebno_dbs, batch_size=128, num_target_block_errors=100, max_mc_iter=10000)
    BLER_all['neural-receiver'] = bler.numpy()

    ##################################################################################################################################
    #################################################################################################################################


    #### an instance of the target model (here we finetune all additional layers of the target model and the last three ResNet Blocks of the source model)
    model_tf_plus = E2ESystem2('neural-receiver',training = True)
    model_tf_plus(1, tf.constant(10.0, tf.float32))

    ### loading wights of layers of base mode to similar layers  of target model 

    model_tf_plus.layers[4].layers[0].set_weights(model_nnrx1.get_weights())
    model_tf_plus.layers[4].layers[1].set_weights(model_nnrx2.get_weights())
    model_tf_plus.layers[4].layers[2].set_weights(model_nnrx3.get_weights())
    model_tf_plus.layers[4].layers[3].set_weights(model_nnrx4.get_weights())
    model_tf_plus.layers[4].layers[4].set_weights(model_nnrx5.get_weights())


    #### freezing some layers of target model

    for layer in model_tf_plus.layers[4].layers[:-5]:  
        layer.trainable = False
    
    #### training of target model

    optimizer = tf.keras.optimizers.Adam()
    ## training of the target model
    for i in range(num_training_iterations):
        # Sampling a batch of SNRs
        ebno_db = tf.random.uniform(shape=[], minval=ebno_db_min, maxval=ebno_db_max)
        # Forward pass
        with tf.GradientTape() as tape:
            rate = model_tf_plus(training_batch_size, ebno_db)
            # Tensorflow optimizers only know how to minimize loss function.
            # Therefore, a loss function is defined as the additive inverse of the BMD rate
            loss = -rate
        # Computing and applying gradients        
        weights = model_tf_plus.trainable_weights
        grads = tape.gradient(loss, weights)
        optimizer.apply_gradients(zip(grads, weights))
        # Periodically printing the progress
        if i % 100 == 0:
            print('Iteration {}/{}  Rate: {:.4f} bit'.format(i, num_training_iterations, rate.numpy()), end='\r')

    #Save the weights in a file
    weights = model_plus.get_weights()
    with open('wgts_tf_plus.pkl', 'wb') as f:
        pickle.dump(weights, f)
        
    ## save the weights in a hd5 format
    #model_tf_layer_plus.save_weights('weights_tf_c2c_120kHz',save_format='h5')



    # parameters for evaluation
    ebno_db_min = -5.0
    ebno_db_max = 10.0
    ebno_dbs = np.arange(ebno_db_min, # Min SNR for evaluation
                        ebno_db_max, # Max SNR for evaluation
                        0.5) # Step





    ####
    # run an instance of the model 
    model_tf_plus = E2ESystem2('neural-receiver')

    # Run one inference to build the layers and loading the weights from the target model
    model_tf_plus(1, tf.constant(10.0, tf.float32))
    with open('wgts_tf_plus.pkl', 'rb') as f:
        weights = pickle.load(f)
    model_tf_plus.set_weights(weights)
    #model_layer_plus.load_weights("/home/es.aau.dk/mw88bt/weights_tf_b2a")



    BLER_plus={}

    # Evaluations of the target model
    _,bler = sim_ber(model_tf_plus, ebno_dbs, batch_size=128, num_target_block_errors=100, max_mc_iter=10000)
    BLER_plus['neural-receiver'] = bler.numpy()

        
                
                


        
    ################################################################################################################################################
    ################################################################################################################################################    
        
        
    #### an instance of the target model (here we only perform feature extraction(FE) and so finetune onlythe newly added targt layers)
    model_tf_fe = E2ESystem2('neural-receiver',training = True)
    model_tf_fe(1, tf.constant(10.0, tf.float32))

    ### loading wights of layers of base mode to similar layers  of target model 

    model_tf_fe.layers[4].layers[0].set_weights(model_nnrx1.get_weights())
    model_tf_fe.layers[4].layers[1].set_weights(model_nnrx2.get_weights())
    model_tf_fe.layers[4].layers[2].set_weights(model_nnrx3.get_weights())
    model_fe.layers[4].layers[3].set_weights(model_nnrx4.get_weights())
    model_tf_fe.layers[4].layers[4].set_weights(model_nnrx5.get_weights())


    #### freezing some layers of target model

    for layer in model_tf_fe.layers[4].layers[:-2]:  
        layer.trainable = False
    
    #### training of target model

    optimizer = tf.keras.optimizers.Adam()
    ## training of the target model
    for i in range(num_training_iterations):
        # Sampling a batch of SNRs
        ebno_db = tf.random.uniform(shape=[], minval=ebno_db_min, maxval=ebno_db_max)
        # Forward pass
        with tf.GradientTape() as tape:
            rate = model_tf_fe(training_batch_size, ebno_db)
            # Tensorflow optimizers only know how to minimize loss function.
            # Therefore, a loss function is defined as the additive inverse of the BMD rate
            loss = -rate
        # Computing and applying gradients        
        weights = model_tf_fe.trainable_weights
        grads = tape.gradient(loss, weights)
        optimizer.apply_gradients(zip(grads, weights))
        # Periodically printing the progress
        if i % 100 == 0:
            print('Iteration {}/{}  Rate: {:.4f} bit'.format(i, num_training_iterations, rate.numpy()), end='\r')

    # Save the weights in a file
    weights = model_tf_fe.get_weights()
    with open(model_weights_path, 'wb') as f:
        pickle.dump(weights, f)
        
    ## save the weights in a hd5 format
    #model_tf_layer_plus_fe.save_weights('weights_tf_c2c_120kHz_fe',save_format='h5')



    # parameters for evaluation
    ebno_db_min = -5.0
    ebno_db_max = 10.0
    ebno_dbs = np.arange(ebno_db_min, # Min SNR for evaluation
                        ebno_db_max, # Max SNR for evaluation
                        0.5) # Step





    ####
    # run an instance of the model 
    model_tf_fe = E2ESystem2('neural-receiver')

    # Run one inference to build the layers and loading the weights from the target model
    model_tf_fe(1, tf.constant(10.0, tf.float32))

    with open(model_weights_path, 'rb') as f:
        weights = pickle.load(f)
    model_tf_fe.set_weights(weights)
    #model_layer_plus_fe.load_weights("/home/es.aau.dk/mw88bt/weights_tf_b2a_fe")




    BLER_fe={}

    # Evaluations of the target model
    _,bler = sim_ber(model_tf_fe, ebno_dbs, batch_size=128, num_target_block_errors=100, max_mc_iter=10000)
    BLER_fe['neural-receiver'] = bler.numpy()

    return BLER_all, BLER_plus, BLER_fe