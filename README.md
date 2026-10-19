# 🚘 Indian License Plate Detection & Recognition — PP-OCRv4 vs PP-OCRv5

<p align="center">
  <b>Computer Vision | Deep Learning | OCR | Model Evaluation</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Framework-PaddlePaddle-red" alt="PaddlePaddle">
  <img src="https://img.shields.io/badge/OCR-PP--OCRv4%20%7C%20PP--OCRv5-orange" alt="OCR Models">
  <img src="https://img.shields.io/badge/Platform-Google%20Colab-yellow?logo=googlecolab" alt="Google Colab">
  <img src="https://img.shields.io/badge/Task-License%20Plate%20Recognition-success" alt="Task">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

## 📌 Project Overview

This project implements a computer vision pipeline for Indian vehicle license plate text recognition using **PP-OCRv4** and **PP-OCRv5**.

The objective is to train, evaluate, export, and compare OCR recognition models on license plate crops, with an emphasis on reproducibility, inference performance, and practical deployment.

The project explores:

- License plate text recognition from cropped vehicle images.
- Fine-tuning and evaluation of PP-OCRv4 and PP-OCRv5 recognition models.
- Character-level decoding using Connectionist Temporal Classification (CTC).
- Batch inference and prediction export to CSV.
- Visual comparison of recognition outputs.
- Error analysis to identify incorrect or incomplete predictions.

**Important scope distinction:** Text recognition reads characters from an image crop. Detecting and localizing a license plate in a full vehicle image is a separate object-detection task and requires an appropriate detector.

## 🔥 Project Tasks

### 🧠 Task 1: PP-OCRv4 License Plate Recognition

PP-OCRv4 is used as a text-recognition baseline and fine-tuning candidate for Indian vehicle registration plates.

The pipeline covers:

- Image preprocessing and resizing.
- Recognition model inference.
- CTC decoding and character dictionary handling.
- Validation and prediction analysis.
- Exporting the trained model for inference.

### 🚀 Task 2: PP-OCRv5 License Plate Recognition

PP-OCRv5 is evaluated using the same general recognition workflow to compare its predictions with PP-OCRv4.

The workflow includes:

- Model loading and inference.
- Recognition output decoding.
- Batch prediction generation.
- Comparison of output strings.
- Analysis of character substitutions, insertions, and deletions.

### 📊 Task 3: Model Comparison and Evaluation

Both models are evaluated on the available test images and recognition dataset.

The comparison focuses on:

- Predicted license plate text.
- Exact-match accuracy when verified ground-truth labels are available.
- Character Error Rate (CER).
- Normalized Edit Distance (where applicable).
- Confidence scores and inference time, when reliably available.
- Qualitative analysis of recognition errors.

Metrics must be computed on a documented, held-out dataset with verified labels. Predictions from a single image are not sufficient to establish overall model accuracy.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Implementation |
| PaddlePaddle | Deep learning inference and training |
| PP-OCRv4 | Text recognition |
| PP-OCRv5 | Text recognition |
| OpenCV | Image preprocessing |
| NumPy | Numerical operations |
| Matplotlib | Visualization |
| Pandas | Prediction tables and CSV export |
| Google Colab | Notebook execution |
| Git and GitHub | Version control and documentation |

## 📓 Google Colab Notebooks

Both notebooks can be opened and executed in Google Colab.

| Notebook | Description |
|---|---|
| **PP-OCRv4 & PP-OCRv5 Training** | [Open Training Notebook](https://colab.research.google.com/drive/1HieV8EkfRyCWFrXjxbMrCB1EJcTA4cK7?usp=sharing) |
| **License Plate Recognition Inference** | [Open Inference Notebook](https://colab.research.google.com/drive/1SF1PhxyTRIy6kfhBgYHUHLT-yoX5rgA6?usp=sharing) |

**How to use:**
1. Open the required notebook using the links above.
2. Sign in to your Google account if prompted.
3. Ensure the notebook has the required access permissions.
4. Follow the notebook cells in order to run the training or inference workflow.

## 📁 Project Directory Structure and Data Pipeline

The project follows a step-by-step pipeline, starting with the original vehicle images and progressing through plate cropping, dataset preparation, model training, and inference.

```text
NUMber_PLate/
│
├── Imagedataset/
│   └── Original vehicle images collected for the project
│
├── plate_crops/
│   └── Plate-region crops generated or collected during
│       the initial cropping stage
│
├── plate_crops_auto/
│   └── Automatically generated license plate crops
│       extracted from the original vehicle images
│
├── processed/
│   └── Processed data and annotation review files,
│       including final_label_review.csv
│
├── recognition_dataset/
│   ├── rec_gt_train.txt
│   └── rec_gt_val.txt
│
├── models/
│   ├── license_plate_ppocrv4/
│   │   └── inference/
│   │       └── Exported PP-OCRv4 inference model files
│   │
│   └── license_plate_ppocrv5/
│       └── inference/
│           └── Exported PP-OCRv5 inference model files
│
├── Video/
│   └── Images used for additional inference experiments
│       and qualitative testing
│
└── license_plate_inference_results.csv
    └── CSV containing inference outputs for comparison
        between PP-OCRv4 and PP-OCRv5
```

### 🔄 How the Directories Were Generated

**1. `Imagedataset/` — Original Data Collection**

Contains the original vehicle images collected for the license plate recognition task. These images serve as the starting point for the data preparation pipeline.

**2. `plate_crops/` — Plate Cropping Stage**

Used to store license plate crops generated or collected during the initial cropping stage. These crops represent the license plate regions intended for text recognition.

**3. `plate_crops_auto/` — Automatic Crop Generation**

Contains automatically generated plate crops obtained from the original vehicle images. These crops provide image inputs for the subsequent recognition and evaluation stages.

**4. `processed/` — Data Processing and Label Review**

Stores intermediate or processed files used to review annotations and prepare the recognition dataset. The `final_label_review.csv` file is used for reviewing the plate labels.

**5. `recognition_dataset/` — Recognition Dataset Preparation**

Contains the text annotation files used by the recognition pipeline:

- `rec_gt_train.txt` — Training samples and their corresponding text labels.
- `rec_gt_val.txt` — Validation samples and their corresponding text labels.

These files connect each plate-crop image with its expected text transcription.

**6. `models/` — Trained and Exported Models**

Contains the exported inference models for PP-OCRv4 and PP-OCRv5. The inference directories store the model configuration and parameter files required to load the respective models for prediction.

**7. `Video/` — Additional Test Images**

Contains images used for additional inference experiments and qualitative testing beyond the primary crop dataset.

**8. `license_plate_inference_results.csv` — Prediction Results**

Stores the inference outputs produced during the comparison of PP-OCRv4 and PP-OCRv5 on the plate-crop dataset. The file supports prediction review and qualitative comparison.

### 🧪 End-to-End Workflow

```text
Original Vehicle Images
          │
          ▼
License Plate Cropping
          │
          ▼
Automatic Plate Crops
          │
          ▼
Label Review and Dataset Preparation
          │
          ▼
Training and Validation Annotation Files
          │
          ▼
PP-OCRv4 / PP-OCRv5 Model Training
          │
          ▼
Exported Inference Models
          │
          ▼
Inference on Plate Crops and Additional Images
          │
          ▼
CSV Results and Prediction Comparison
```

**Note:** The directory descriptions above reflect the observed project structure and intended pipeline. The exact generation method for every file should only be claimed where it is confirmed by the corresponding notebook or processing code.
*Dataset files and model weights are not necessarily included in this repository. Follow the dataset preparation instructions before running the notebooks.*

## 📂 Dataset Access

The complete dataset is hosted on Google Drive because the image files exceed the practical size limits for uploading the entire dataset directly to GitHub.

**[📥 Access the Dataset on Google Drive](https://drive.google.com/drive/folders/1ssk2MllfyFtpC9PzH3L9xZz5uc-jItgD?usp=sharing)**

The dataset contains the original vehicle images, license plate crops, and related data-preparation files used in the recognition workflow.

### How to Use the Dataset

1. Open the Google Drive link above.
2. Download the required dataset folders.
3. Upload or mount the dataset in Google Colab.
4. Update the dataset paths in the notebook if required.
5. Run the notebook cells in the specified order.

**Note:** Access depends on Google Drive sharing permissions. Ensure that the reviewer can access the folder. Avoid including private information or files unrelated to the assignment.

## 📂 Dataset Preparation

The recognition pipeline requires cropped license plate images and their corresponding text labels.

A recommended dataset structure is:

```text
license_plate_dataset/
│
├── images/
│   ├── plate_001.jpg
│   ├── plate_002.jpg
│   └── ...
│
├── labels.tsv
└── DATASET_NOTES.md
```

The `labels.tsv` file should contain two tab-separated fields per line, without a header:

```text
images/plate_001.jpg    MH01CR7388
images/plate_002.jpg    DL8CAF1234
```

Replace the example labels with the actual verified text corresponding to each image.

### Dataset guidelines

- Use at least 100 distinct plate crops if required by the assignment.
- Verify every transcription against the source image.
- Remove duplicates and invalid images.
- Keep training and validation splits separate.
- Use a reproducible split seed.
- Document the dataset source, license, collection date, and annotation process.
- Avoid publishing identifiable plate images unless their use and distribution are permitted.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/shubh-lyn/COMPUTER_VISION_ASSIGNMNET.git
cd COMPUTER_VISION_ASSIGNMNET
```

### 2. Open the notebooks

Open the notebooks in Google Colab:

- `PPOCR_License_Plate_Training.ipynb`
- `PPOCR_License_Plate_Inference.ipynb`

For training, select an available GPU runtime when possible.

### 3. Configure the dataset paths

Update the dataset and output paths in the notebooks to match your environment.

For example:

```python
DATASET_ROOT = "/content/drive/MyDrive/license_plate_dataset"
```

The exact variable names depend on the notebook configuration.


### 4. Install dependencies

Follow the installation cell in the notebook and use a PaddlePaddle build compatible with the active Python and CUDA environment.

Do not mix incompatible CPU and GPU PaddlePaddle installations.

## 🏋️ Model Training Workflow

The training workflow is designed to cover:

1. Dataset and label validation.
2. Image preprocessing.
3. Reproducible training/validation split.
4. Model configuration and checkpoint loading.
5. Fine-tuning PP-OCRv4 and PP-OCRv5.
6. Loss and validation metric tracking.
7. Model checkpoint selection.
8. Exporting inference models.
9. Saving logs and evaluation artifacts.

**Note:** PP-OCRv4 and PP-OCRv5 use different configurations and character dictionaries. Their checkpoints and dictionaries must not be interchanged.

## 🔍 Inference Workflow

The inference notebook is intended to:

1. Load the exported recognition models.
2. Read license plate crop images.
3. Resize and normalize the input according to the selected model's configuration.
4. Run inference using the exported model.
5. Decode model output using the matching character dictionary and CTC rules.
6. Compare PP-OCRv4 and PP-OCRv5 predictions.
7. Export prediction results to CSV.
8. Inspect incorrect and incomplete recognitions.

Example of the direct PaddlePaddle inference interface:

```python
from paddle.inference import Config, create_predictor

model_dir = "models/license_plate_ppocrv4/inference"

config = Config(
    f"{model_dir}/inference.json",
    f"{model_dir}/inference.pdiparams"
)

config.disable_gpu()

predictor = create_predictor(config)

print("Model inputs:", predictor.get_input_names())
print("Model outputs:", predictor.get_output_names())
```

The example assumes the exported model files are present at the specified path. Input preprocessing and output decoding must match the exported model's configuration.

## 📊 Results and Evaluation

### Qualitative single-image test

One manually inspected test image was used to verify that the exported models could produce decoded text.

| Model | Decoded prediction | Observation |
|---|---|---|
| PP-OCRv4 | `MH.01.CR.7388` | Matches the visual reference `MH01CR7388` after removing punctuation |
| PP-OCRv5 | `901.CR-73880` | Incorrect characters and extra digits |

The plate text above is a visual reference for this specific example, not a dataset-wide evaluation result.

### Quantitative evaluation

| Metric | Status |
|---|---|
| Exact-match accuracy | Requires evaluation on verified ground-truth labels |
| Character Error Rate (CER) | Requires verified reference transcriptions |
| Normalized Edit Distance | Report only when computed on the evaluation set |
| Mean inference time | Requires repeatable timing measurements |
| Dataset size and split | Record the actual counts produced by the notebook |

No overall accuracy, mAP, precision, or recall values are claimed here without the corresponding validated evaluation. Object-detection metrics such as mAP require plate bounding-box annotations and a detector.

## 🚧 Challenges Addressed

- Differences between PP-OCRv4 and PP-OCRv5 character dictionaries.
- Correct CTC blank-token handling and repeated-character decoding.
- Image resizing and normalization requirements.
- Recognition errors caused by poor crop quality.
- Distinguishing recognition from full-image plate detection.
- Dependency compatibility across Python and PaddlePaddle versions.
- Evaluating models without overstating results from a small dataset.

## 🔮 Future Improvements

- Expand the dataset with diverse Indian registration plate formats.
- Improve label quality through manual verification.
- Evaluate both models on the same held-out test set.
- Add CER, exact-match accuracy, and inference-time reports.
- Integrate a dedicated license plate detector for full vehicle images.
- Add confidence-aware prediction filtering.
- Optimize exported models for CPU and GPU inference.
- Explore deployment on NVIDIA Jetson edge devices.
- Add reproducible experiment logs and automated smoke tests.

## 🎯 Applications

- Vehicle registration plate recognition.
- Parking and access-control systems.
- Traffic monitoring research.
- Smart transportation prototypes.
- Computer vision and OCR benchmarking.
- Edge AI experimentation.

Applications involving public-road or surveillance imagery should comply with applicable privacy laws and data-use restrictions.

## 📌 Conclusion

This project explores the training and inference workflow for Indian license plate text recognition using PP-OCRv4 and PP-OCRv5.

It covers dataset preparation, model export, direct inference, character decoding, batch processing, and qualitative comparison. The single-image test demonstrates that PP-OCRv4 can produce a matching decoded string on the inspected example, while PP-OCRv5 produces an incorrect result on that sample.

A larger, independently labeled evaluation set is required before drawing conclusions about overall model performance.

## 📄 License

This project is intended to be released under the MIT License. Before distributing it, include a `LICENSE` file containing the full MIT License text and verify that all third-party code, model weights, and datasets permit the intended use.

## 👤 Author

**Achyut Kumar Pandey**

Computer Vision | AI/ML | Cybersecurity

GitHub: [@shubh-lyn](https://github.com/shubh-lyn)

---

*This repository is intended for educational and research purposes. Model performance depends on dataset quality, annotation accuracy, preprocessing, and evaluation methodology.*
