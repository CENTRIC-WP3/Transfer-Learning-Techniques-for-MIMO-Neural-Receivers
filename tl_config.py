#from model_transfer import model_transfer
from FineTuning import fine_tuning
#from FineTuning_Site_Specific import FineTuning_Site_Specific
#from mamel_metal import mamel_metal
#from rtile_metal import rtile_metal
from plot_fig import plot_fig
from no_TL import without_tl
from model_tf import model_transfer
import pickle
import os

####################################################################
############### WITHOUT TL  #####################################

chann_model = "C" # channel model for the nueral receiver . Optons are ("A","B","C","D", "E","UMi","UMa")
subc_space =15e3 # OFDM subcarrier spacing. Optons are (15e3, 30e3,60e3,120e3)
alf = 0.0001 # iteration weight. Option could be from 0.001 - 1
bit_per_sys= 2  # number of bits per symbols. Options are (2 for QPSK, 4 for 16 QAM and 6 for 6QAM)

wtl_bler =  without_tl(chann_model, subc_space, alf,bit_per_sys)
    #file_save = scenario + "_eval_" + target_model
with open("bler_noTL.pkl", 'wb') as f:
    pickle.dump(wtl_bler, f)

####################################################################
############### MODEL TRANSFER #####################################
chann_model = "C" # channel model for the nueral receiver . Optons are ("A","B","C","D", "E","UMi","UMa")
subc_space =15e3 # OFDM subcarrier spacing. Optons are (15e3, 30e3,60e3,120e3)
bit_per_sys= 2  # number of bits per symbols. Options are (2 for QPSK, 4 for 16 QAM and 6 for 6QAM)
alf = 0.0001 # iteration weight. Option could be from 0.001 - 1
source_filepath ="/home/es.aau.dk/mw88bt/weights_BL" # path of the weight file of the source model
source_scenario ="C" # scenario could be channel model("A","B","C","D", "E","UMi","UMa"), subcarrier spacing (15e3, 30e3,60e3,120e3) or modulation scheme(2 for QPSK, 4 for 16 QAM and 6 for 6QAM)
target_scenario = "UMi" # scenario could be channel model("A","B","C","D", "E","UMi","UMa"), subcarrier spacing (15e3, 30e3,60e3,120e3) or modulation scheme(2 for QPSK, 4 for 16 QAM and 6 for 6QAM)
eval_bler =  model_transfer(chann_model,subc_space,bit_per_sys,source_filepath, source_scenario,target_scenario, alf)
    #file_save = scenario + "_eval_" + target_model
with open(" sorce_scenario + '_to_' + 'target_scenario' .pkl", 'wb') as f:
    pickle.dump(eval_bler, f)

####################################################################
############### FINETUNING & FEATURE EXTRACTION##########
    
chann_model = "C" # channel model for the nueral receiver . Optons are ("A","B","C","D", "E","UMi","UMa")
subc_space =15e3 # OFDM subcarrier spacing. Optons are (15e3, 30e3,60e3,120e3)
bit_per_sys= 2  # number of bits per symbols. Options are (2 for QPSK, 4 for 16 QAM and 6 for 6QAM)
alf = 0.0001 # iteration weight. Option could be from 0.001 - 1
source_filepath ="C:\\Users\\MW88BT\\OneDrive - Aalborg Universitet\\my_vscode_scripts\\Comprehns_Evaluation\\weights_BL" # path of the weight file of the source model
source_scenario ="C" # scenario could be channel model("A","B","C","D", "E","UMi","UMa"), subcarrier spacing (15e3, 30e3,60e3,120e3) or modulation scheme(2 for QPSK, 4 for 16 QAM and 6 for 6QAM)
target_scenario = "umi" # scenario could be channel model("A","B","C","D", "E","UMi","UMa"), subcarrier spacing (15e3, 30e3,60e3,120e3) or modulation scheme(2 for QPSK, 4 for 16 QAM and 6 for 6QAM)
## to run the finetuning case alone 
all_bler, plus_bler, fe_bler =  fine_tuning(chann_model ,subc_space, bit_per_sys, alf,source_scenario, source_filepath)
    #file_save = scenario + "_eval_" + target_model
fname_all = source_scenario + '_finetune_' + target_scenario 
fname_plus = source_scenario + '_finetune_plus_' + target_scenario 
fname_fe = source_scenario + '_fe_' + target_scenario 
with open(fname_all, 'wb') as f:
    pickle.dump(all_bler, f)
with open(fname_plus, 'wb') as f:
    pickle.dump(plus_bler, f)
with open(fname_fe, 'wb') as f:
    pickle.dump(fe_bler, f)

####################################################################
################## plotting the BER/BLER ##################
 """   
plot_file_path_1 =""
plot_file_path_2 =""
plot_file_path_3 =""
plot_file_path_4 =""
plot_file_path_5 =""
plot_file_path_6 =""
plot_markers = "" # list the markers for each plot (s, d, *, <, >)
list_label = "" # list the labels for each the plots 
list_colour ="" # list the colours of each of the plots (C1, C2, .....)
ebno_dbs = "" # a range of snr, same as in simulations 
plot_files = ['plot_file_path_1', 'plot_file_path_2', 'plot_file_path_3']
plot_fig(plot_files,ebno_dbs, plot_markers,list_label,list_colour)

###################################################################
#######################################################################

"""