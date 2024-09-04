## Introduction 
Here in this repository are the codes for the paper - *On transfer Learning for A Fully Convolutional Deep Neural SIMO Receiver*. The paper evaluates the performance of two fine tuning based techniques and a feature extraction technique for a deep neural receiver.

## Description (Simulation Environment and AI Algorithm)​
### Dependencies
- Python
- Sionna
- Tensorflow
- Keras

### Code Description
1. The main configuration file is tl_config.py from which function files can be called and prameters set to run to run eperiments.
2.  Finetuning.py implements finetuning, finetuning+ and feature extraction.
3.  Without TL implements the classical machine learning withot transfer learning.
4.  model_transfer simply evaluates a pretrained model on target dataset.
5.  plot_fig.py  plots the BLERs resulting from running the experiments.
6.  The jupiter notebook figures_notebook)  can be used to interactively visualize results.

## Potential Application
This repository provides a framework for testing transfer learnign algorithms on a deep neural  SIMO receiver.
## Example Usage
Training deep neural networks typically rewuires large datasets, and where configurations are dynamic as in wireless communication, new datasets may neet to be generated for each confiration. Transfer learning solves the large dataset problem by reusing experience from a source model trained on a large dataset.
### Simulation Environment
|Parameter|Value|
| --- | --- |
|Number of Symbols | $14 $|
|Number of Subcarries  |$128$|
|Guard carriers index | [5, 6]|
|subcarrier spacing| 30, 60 and 120 KHz|
|Pilots symbols index|2 ,11] |
|Channel Models| 3GPP UMi, CDL –A, B, C, D and E|
|Modulation| QPSK, 16QAM, 64QAM|
|Coding| LDPC|
|Batch size|$128$|
|Optimizer|Adam|
|Input Shape|$M \times N_{rx} \times F \times S$|
|Output Shape|$M \times n_{bits}$|
|Number of Transmitters|1|
|Number of Transmit Antennas|1|
|Number of Receivers|1|
|Receive Antenna Configuration|$1\times1$|
| Direction| Uplink |
|Source dataset size|$3, 480, 000$|
|Target dataset size| $348, 000$|
|Signal-to-noise-ratio range|[-4 - 8] dB|
|Learning rate| $10^{-3}$|

### Running an Experiment
An experiment can easily be run by calling a function, like finetunng and specifying the parameters like sucarrier spacing, bits per symbol etc
```
chann_model = "" # channel model for the nueral receiver . Options are ("A","B","C","D", "E","UMi","UMa")
subc_space = # OFDM subcarrier spacing. Optons are (15e3, 30e3,60e3,120e3)
bit_per_sys=  # number of bits per symbols. Options are (2 for QPSK, 4 for 16 QAM and 6 for 6QAM)
alf = # iteration weight. Option could be from 0.001 - 1
source_filepath ="" # path of the weight file of the source model
source_scenario ="" # scenario could be channel model("A","B","C","D", "E","UMi","UMa"), subcarrier spacing (15e3, 30e3,60e3,120e3) or modulation scheme(2 for QPSK, 4 for 16 QAM and 6 for 6QAM)
target_scenario = "umi" # scenario could be channel model("A","B","C","D", "E","UMi","UMa"), subcarrier spacing (15e3, 30e3,60e3,120e3) or modulation scheme(2 for QPSK, 4 for 16 QAM and 6 for 6QAM)

all_bler, plus_bler, fe_bler =  fine_tuning(chann_model ,subc_space, bit_per_sys, alf,source_scenario, source_filepath)
```

### Results

<img src="https://github.com/user-attachments/assets/9fc66c1c-55bd-419b-b2f0-f1d9a884d99b" width="500">
<img src="https://github.com/user-attachments/assets/e87b7d80-2285-491e-b549-4685a5855bae" width="500">




## Citation
```bibtex
@misc{uyoata2024transferlearningfullyconvolutional,
      title={On Transfer Learning for a Fully Convolutional Deep Neural SIMO Receiver}, 
      author={Uyoata E. Uyoata and Ramoni O. Adeogun},
      year={2024},
      eprint={2408.16401},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2408.16401}, 
}
```


