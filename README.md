# Github repo supporting paper
### Deep Convolutional Neural Network Applied to Electroencephalography: Raw Data vs Spectral Features
(On arXiv: https://arxiv.org/abs/2105.04762)

## Code layout
- `environment.yml`: conda environment set up
- `HBN_NEMAR_Pheno.csv`: Subjects information
- `GSN_HydroCel_129.sfp`: Channels standard position file provided by HBN
- `restingstate_prepare_clean_master.m`: Preprocess data
- `load_data_master.m`: Take result from `restingstate_prepare_clean_master.m` and sub-select channel to prepare final raw and spectral data. 
- `SexPrediction-Original-Raw.ipynb`: Notebook to train referenced work model on raw data (R-SCNN)
- `SexPrediction-Original-Topo.ipynb`: Notebook to train referenced work model on spectral data (S-SCNN)
- `SexPrediction-VGG-Raw.ipynb`: Notebook for training repurposed VGG-16 model on raw data (R-VGG)
- `SexPrediction-VGG-Topo.ipynb`: Notebook for training repurposed VGG-16 model on spectral data (S-VGG)
