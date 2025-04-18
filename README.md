# SpatialOCT-Glaucoma

This repository contains the implementation of the paper entitled ["Spatial-aware Transformer-GRU Framework for Enhanced Glaucoma Diagnosis from 3D OCT Imaging."](https://doi.org/10.1109/JBHI.2025.3550394)

Please contact ashtari.mona@gmail.com for further information.

## Project Structure
```plaintext
project/
│
├── OCT_data
│
├── RNN_main.py # Main script for RNN sequential processing
│
├── feature_extraction_main.py # Main script for feature extraction
│
├── RETFound_oct_weights.pth
│
└── split_index.pickle # Indices for cross-validation
```

## Setup
1. Download the RETFound pre-trained weights for OCT data from [RETFound_MAE](https://github.com/rmaphoh/RETFound_MAE) GitHub and place it in the project folder.

2. Download the 3D OCT dataset from [here](https://zenodo.org/records/1481223) and extract it. Place the extracted images into the OCT_data folder.

## Usage
3. Navigate to the project directory and execute the feature extraction script:
```
python feature_extraction_main.py
```

4. Execute the RNN sequential processing script:
```
python RNN_main.py
```

## Citation
If you use this repository for your research, please cite our paper:
```
@article{ashtari2024spatial,
  title={Spatial-aware Transformer-GRU Framework for Enhanced Glaucoma Diagnosis from 3D OCT Imaging},
  author={Ashtari-Majlan, Mona and Dehshibi, Mohammad Mahdi and Masip, David},
  journal={arXiv preprint arXiv:2403.05702},
  year={2024}
}
```

