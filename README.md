# SpatialOCT-Glaucoma

This repository contains the implementation of the paper entitled "Spatial-aware Transformer-GRU Framework for Enhanced Glaucoma Diagnosis from 3D OCT Imaging." The paper is available on arXiv: [https://arxiv.org/abs/2403.05702](https://arxiv.org/abs/2403.05702).

Please contact ashtari.mona@gmail.com for further information.

## Project Structure
```plaintext
project/
│
├── models/
│   └── RNN_model.py
│
├── vit_model/
│   └── models_vit.py
│
├── OCT_data/
│   └── class0
│   └── class1
│
├── utils/
│   ├── OCT_dataset.py
│   ├── data_preparation.py
│   ├── evaluation.py
│   ├── feature_dataset.py
│   ├── pos_embed.py
│   ├── train.py
│   └── utils.py
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
### Data Preparation
1. Download the OCT dataset from [Zenodo](https://zenodo.org/records/1481223) and extract it.
2. Separate the data into two classes and place them into the `OCT_data/class0` and `OCT_data/class1` folders.
3. Download the pre-trained RETFound weights on OCT data from [Google Drive](https://drive.google.com/file/d/1m6s7QYkjyjJDlpEuXm7Xp3PmjN-elfW2/view?usp=sharing) and place it in the project folder.


## Usage
To run the feature extraction script, navigate to the project directory and execute:
python feature_extraction_main.py

To perform RNN sequential processing, run:
python RNN_main.py

## Citation
If you use this repository for your research, please cite our paper:
@article{ashtari2024spatial,
  title={Spatial-aware Transformer-GRU Framework for Enhanced Glaucoma Diagnosis from 3D OCT Imaging},
  author={Ashtari-Majlan, Mona and Dehshibi, Mohammad Mahdi and Masip, David},
  journal={arXiv preprint arXiv:2403.05702},
  year={2024}
}

