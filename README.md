# Lung Disease Classification from Chest X-Ray Images

A machine learning project for **multi-class classification of chest X-ray images** into:

- **Normal**
- **COVID-19**
- **Tuberculosis (TB)**

The project is inspired by the research paper:

> **A New Approach Method for Multi Classification of Lung Diseases using X-Ray Images**  
> International Journal of Advanced Computer Science and Applications (IJACSA), Vol. 14, No. 7, 2023.

The goal is to reproduce the general machine-learning workflow used in the paper using publicly available chest X-ray datasets.

---

## Project Overview

This project applies traditional machine-learning algorithms to chest X-ray images.

The general pipeline is:

```text
Chest X-Ray Images
        ↓
Class Selection
        ↓
Balanced Sampling
        ↓
Image Resizing
        ↓
Train/Test Split
        ↓
Data Augmentation
        ↓
Flattening
        ↓
Machine Learning Models
        ↓
Evaluation
```

The following models are evaluated:

- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest
- Extra Trees

The models are compared using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## Dataset Sources

This project combines images from **two public Kaggle datasets**.

### 1. COVID-19 Radiography Database

Source:

https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

Used classes:

- `COVID`
- `Normal`

The downloaded dataset contains multiple categories such as:

```text
COVID
Lung_Opacity
Normal
Viral Pneumonia
```

For this project, only **COVID** and **Normal** are used.

The `images/` directories are used for classification. The corresponding `masks/` directories are not used because this project focuses on **image classification**, not lung segmentation.

---

### 2. Tuberculosis (TB) Chest X-ray Database

Source:

https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset

Used class:

- `Tuberculosis`

The TB dataset also contains Normal images, but the project currently uses the Normal class from the COVID-19 Radiography Database to keep the data preparation workflow simple and consistent.

---

## Local Dataset Distribution

The downloaded data contained:

| Class | Available Images |
|---|---:|
| Normal | 10,192 |
| COVID-19 | 3,616 |
| Tuberculosis | 700 |

Because the classes are highly imbalanced, the current implementation uses **random undersampling** to create an initial balanced dataset:

| Class | Selected Images |
|---|---:|
| Normal | 700 |
| COVID-19 | 700 |
| Tuberculosis | 700 |
| **Total** | **2,100** |

A fixed random seed (`42`) is used to make the sampling reproducible.

---

## Dataset Structure

The raw datasets are expected to be placed inside:

```text
data/raw/
```

Example structure:

```text
data/
└── raw/
    ├── COVID-19_Radiography_Dataset/
    │   ├── COVID/
    │   │   ├── images/
    │   │   └── masks/
    │   ├── Normal/
    │   │   ├── images/
    │   │   └── masks/
    │   ├── Lung_Opacity/
    │   └── Viral Pneumonia/
    │
    └── TB_Chest_Radiography_Database/
        ├── Normal/
        └── Tuberculosis/
```

The project reads images from:

```text
data/raw/COVID-19_Radiography_Dataset/COVID/images
data/raw/COVID-19_Radiography_Dataset/Normal/images
data/raw/TB_Chest_Radiography_Database/Tuberculosis
```

The raw datasets are intentionally excluded from Git using `.gitignore`.

---

## Preprocessing

Each image is:

1. Loaded using OpenCV.
2. Resized to:

```text
100 × 100 pixels
```

3. Stored as a NumPy array.
4. Assigned a numeric label:

```text
Normal       -> 0
COVID-19     -> 1
Tuberculosis -> 2
```

After preprocessing:

```text
X shape: (2100, 100, 100, 3)
y shape: (2100,)
```

---

## Train/Test Split

The dataset is divided into:

- **80% training**
- **20% testing**

Using stratified splitting keeps the three classes equally represented.

Expected distribution:

```text
Training:
560 Normal
560 COVID
560 Tuberculosis

Testing:
140 Normal
140 COVID
140 Tuberculosis
```

Shapes:

```text
X_train: (1680, 100, 100, 3)
X_test:  (420, 100, 100, 3)
```

---

## Data Augmentation

Data augmentation is applied **only to the training set** to avoid data leakage.

The augmentation configuration follows the general settings used in the reference paper:

```python
rotation_range=40
shear_range=0.2
zoom_range=0.2
width_shift_range=0.2
height_shift_range=0.2
horizontal_flip=True
```

One augmented image is generated for every original training image.

Therefore:

```text
Original training images: 1680
Augmented images:         1680
Final training images:    3360
```

The test set remains unchanged.

---

## Feature Preparation

Traditional machine-learning algorithms expect tabular feature vectors rather than 3D image arrays.

Each image:

```text
100 × 100 × 3
```

is flattened into:

```text
30,000 features
```

Resulting shapes:

```text
Training: (3360, 30000)
Testing:  (420, 30000)
```

Pixel values can also be normalized from:

```text
0 - 255
```

to:

```text
0 - 1
```

Normalization is particularly useful for distance-based models such as KNN and SVM.

---

## Models

### Extra Trees

```python
ExtraTreesClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)
```

### Random Forest

```python
RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)
```

### K-Nearest Neighbors

```python
KNeighborsClassifier(
    n_neighbors=3
)
```

### Support Vector Machine

```python
SVC(
    kernel="linear"
)
```

---

## Evaluation Metrics

Each model is evaluated using:

```text
Accuracy
Precision
Recall
F1-score
Confusion Matrix
Classification Report
```

A final comparison table is generated to compare all trained models.

Example:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Extra Trees | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD |
| SVM | TBD | TBD | TBD | TBD |
| KNN | TBD | TBD | TBD | TBD |

Replace `TBD` with the final experimental results after training.

---

## Project Structure

```text
lung-disease-classification/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── 01_lung_disease_classification.ipynb
│
├── results/
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Project

Open the project in VS Code and select the Python interpreter from:

```text
.venv
```

Then open:

```text
notebooks/01_lung_disease_classification.ipynb
```

and run the notebook cells in order.

---

## Important Notes

This project is a **reimplementation of the general methodology**, not an exact reproduction of the original paper.

The public Kaggle datasets used here are different from the hospital dataset used by the paper authors. Therefore, model results should not be directly interpreted as better or worse than the reported results in the paper.

Another limitation is that the disease classes are partially collected from different source datasets. Differences in image acquisition, preprocessing, contrast, resolution, or dataset-specific artifacts could influence model performance.

---

## Medical Disclaimer

This project is intended for **educational and research purposes only**.

It is **not a medical diagnostic system** and must not be used to diagnose COVID-19, tuberculosis, or any other medical condition.

---

## References

### Research Paper

Sri Heranurweni, Andi Kuniawan Nugroho, and Budiani Destyningtias.  
**"A New Approach Method for Multi Classification of Lung Diseases using X-Ray Images."**  
IJACSA, Vol. 14, No. 7, 2023.

### Datasets

**COVID-19 Radiography Database**  
Kaggle:  
https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

**Tuberculosis (TB) Chest X-ray Database**  
Kaggle:  
https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset