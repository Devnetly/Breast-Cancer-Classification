# AI System for Breast Cancer Diagnosis

## Overview

This university project, conducted in 2024, is a collaboration between our team and the Anti-Cancer Center of Sidi Bel Abbes (CAC). The objective of this project was to develop an AI system capable of diagnosing breast cancer through the classification of gigapixel hematoxylin and eosin histopathological images (whole slide images). We used the [BRCAS](https://www.bracs.icar.cnr.it/) dataset for this purpose. The process mainly consists of two stages:

- **Feature Extraction**: This involves processing the high-resolution whole slide images to generate a more compact representation that can be processed by deep learning models.

- **Models Training**: This step involves training the actual models to classify the WSIs, using the compressed WSIs generated from the previous step as input.

The final model was deployed in a desktop application where a WSI can be selected, then the feature extraction step is applied before passing the result to the model for inference and outputting the probabilities of each class, for more details please refere to : [Report](https://github.com/Devnetly/Breast-Cancer-Detection/blob/main/docs/report.pdf).

<img src="https://github.com/Devnetly/Breast-Cancer-Detection/blob/main/docs/Report/figures/approach.png?raw=true" alt="process overiew" width="100%"/>


## Members

- Abdelnour Fellah: [ab.fellah@esi-sba.dz](mailto:ab.fellah@esi-sba.dz)
- Abderrahmane Benounene: [a.benounene@esi-sba.dz](mailto:a.benounene@esi-sba.dz)
- Adel Abdelkader Mokadem: [aa.mokadem@esi-sba.dz](mailto:aa.mokadem@esi-sba.dz)
- Meriem Mekki: [me.mekki@esi-sba.dz](mailto:me.mekki@esi-sba.dz)
- Yacine Lazreg Benyamina: [yl.benyamina@esi-sba.dz](mailto:yl.benyamina@esi-sba.dz)

## Steup

To set up the project, please follow these steps:

```sh
git git@github.com:Devnetly/Breast-Cancer-Detection.git
cd Breast-Cancer-Detection
conda create -n breast-cancer-detection
conda activate breast-cancer-detection
pip install -r requirements.txt
git clone git@github.com:yiqings/RandStainNA.git
git clone git@github.com:mahmoodlab/CLAM.git
```
## Feature Extractors fine tuning

```sh
    cd src/training/feature_extractors
    python main.py [-h] [--batch-size BATCH_SIZE] [--epochs EPOCHS] \
        [--learning-rate LEARNING_RATE] \
        [--model-type {resnet18,resnet34 resnet50,vit}] \
        --weights-folder WEIGHTS_FOLDER --histories-folder HISTORIES_FOLDER \
        [--preprocessing {nothing,stain-normalization,augmentation,stain-augmentation}] \
        [--sampler {random,balanced}] [--dropout DROPOUT] \
        [--decay-rate DECAY_RATE] [--optimizer {adam,sgd,rmsprop}] \
        [--last-epoch LAST_EPOCH] [--weight-decay WEIGHT_DECAY] \
        [--depth DEPTH] [--num-workers NUM_WORKERS] \
        [--prefetch-factor PREFETCH_FACTOR] \
        [--class-weights CLASS_WEIGHTS] [--momentum MOMENTUM]
```

## Feature Extraction

Apply feature extraction on the dataset using a pre-trained model : 

### Grid Based Feature Extraction


```sh
    cd src/feature_extraction
    python grid.py [-h] --in-path IN_PATH --out-path OUT_PATH \
        [--train TRAIN] [--test TEST] [--val VAL] \
        [--patch-size PATCH_SIZE] [--model MODEL] \
        --model-weights MODEL_WEIGHTS --metadata-path METADATA_PATH \
        [--n N] [--batch-size BATCH_SIZE] \
        [--num-workers NUM_WORKERS] [--prefetch-factor PREFETCH_FACTOR]
```

### Feature Extraction With Patch Selection

```sh
    cd src/feature_extraction
    python vector.py [-h] --coords-path COORDS_PATH --in-path IN_PATH  \
        --out-path OUT_PATH [--train TRAIN] [--test TEST] [--val VAL] \
        [--patch-size PATCH_SIZE] [--model MODEL] \
        --model-weights MODEL_WEIGHTS --metadata-path METADATA_PATH \
        [--n N] [--batch-size BATCH_SIZE] [--num-workers NUM_WORKERS] \ [--prefetch-factor PREFETCH_FACTOR]
```

## Training

To train WSI Classifiers on the compressed dataset : 

```sh
    cd src/training/wsi_classifiers
    python3 main.py [-h] [--model {ABNN,ACMIL,HIPT}] \
        --tensors TENSORS --weights-folder WEIGHTS_FOLDER \
        --histories-folder HISTORIES_FOLDER [--epochs EPOCHS] \
        [--num-workers NUM_WORKERS] [--prefetch-factor PREFETCH_FACTOR] \
        [--last-epoch LAST_EPOCH] [--save-weights-every SAVE_WEIGHTS_EVERY] 
        [--dropout DROPOUT] [--learning-rate LEARNING_RATE] \
        [--weight-decay WEIGHT_DECAY] [--sampler {random,balanced}] \
        [--min-lr MIN_LR] [--decay-alpha DECAY_ALPHA] \
        [--filters-in FILTERS_IN] [--filters-out FILTERS_OUT] \
        [--use-lr-decay USE_LR_DECAY] \
        [--features {resnet18,resnet34,resnet50,vit,hipt_4k}] \
        [--mask-rate MASK_RATE] [--branches-count BRANCHES_COUNT] \
        [--k K] [--d D] [--data-augmentation DATA_AUGMENTATION]
```

## Run the desktop application

To run the desktop app associated with the project :

```sh
    cd deployement
    python app.py
```