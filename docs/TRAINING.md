# Training the Mask Classifier

The classifier in `models/face_mask_detector.h5` was trained with the
workflow in [`notebooks/face_mask_training.ipynb`](../notebooks/face_mask_training.ipynb)
(originally run on Google Colab). The hyper-parameters are mirrored in
`TrainingConfig` (`src/mask_detection/config.py`).

## Dataset

Images of faces in two classes — `with_mask` and `without_mask` — organised
in class-named subfolders (the class label is taken from the parent folder
name). The dataset is **not** tracked in this repository.

Each image is loaded at 224×224 and preprocessed with MobileNetV2's
`preprocess_input`. Labels are one-hot encoded, and the data is split
80 / 20 (stratified, `random_state=42`).

## Model

Transfer learning on **MobileNetV2** (ImageNet weights, `include_top=False`,
input 224×224×3) with all base layers frozen, plus a custom head:

```
MobileNetV2 base (frozen)
 → AveragePooling2D(7×7)
 → Flatten
 → Dense(128, relu)
 → Dropout(0.5)
 → Dense(2, softmax)      # (mask, without_mask)
```

## Training setup

| Hyper-parameter | Value |
| --- | --- |
| Optimizer | Adam, lr = 1e-4, decay = lr / epochs |
| Loss | binary cross-entropy |
| Epochs | 20 |
| Batch size | 32 |

Data augmentation (`ImageDataGenerator`): rotation ±20°, zoom 0.15,
width/height shift 0.2, shear 0.15, horizontal flips, nearest fill.

## Evaluation

The notebook prints a scikit-learn `classification_report` on the held-out
split and plots train/validation loss and accuracy over epochs.

## Reproducing

1. Install training extras: `pip install -e ".[train]"`.
2. Place the dataset under `dataset/with_mask` and `dataset/without_mask`
   and update the dataset path in the notebook (it still contains the
   original Colab path).
3. Run the notebook end-to-end; save the resulting model to
   `models/face_mask_detector.h5`.
