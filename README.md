# Machine-Learning-Based-Arrhythmia-Detection-Using-ECG-Signals-for-Cardiac-Health-Monitoring

## 1. Project Overview

This project developed a **machine learning and deep learning-based framework for classifying cardiac arrhythmias using Electrocardiogram (ECG) data**.

The primary objective was to develop and evaluate multiple machine learning and deep learning models for accurately identifying different types of arrhythmia from ECG-derived features.

The project followed an end-to-end machine learning workflow covering:

* Data loading
* Data preprocessing
* Missing-value imputation
* Feature scaling
* Dimensionality reduction
* Class imbalance handling
* Model training
* Model evaluation
* Ensemble learning
* Performance comparison
* Confusion matrix analysis
* Model visualization

The project evaluated both traditional machine learning algorithms and neural-network-based approaches.

---

## 2. Dataset Processing

### Dataset

The project used an `arrhythmia.csv` dataset containing features extracted from ECG data.

The dataset was loaded using **Pandas** and processed before being supplied to the machine learning models.

### Data Preprocessing

The preprocessing pipeline included:

* Replacement of missing values represented by `?` with `NaN`
* Mean-based missing-value imputation
* Feature standardization using `StandardScaler`
* Dimensionality reduction using **Principal Component Analysis (PCA)**
* Reduction to **100 principal components**
* Class imbalance handling using **SMOTE**
* Stratified train-test splitting
* Label indexing for Keras and TabNet models
* Data reshaping for CNN and LSTM architectures

---

## 3. Prediction Task

The project formulated the problem as a **multi-class classification task**.

The objective was to classify ECG observations into different arrhythmia categories using the available ECG-derived features.

The models were trained on the preprocessed feature representation and evaluated according to their classification performance.

---

## 4. Machine Learning Pipeline

The overall workflow can be represented as:

```text
             ECG / Arrhythmia Dataset
                       │
                       ▼
                 Data Loading
                       │
                       ▼
              Missing Value Handling
                       │
                       ▼
                Feature Scaling
                       │
                       ▼
                     PCA
                       │
                       ▼
                    SMOTE
                       │
                       ▼
              Train/Test Split
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
 Traditional ML                Deep Learning
          │                         │
 ┌────────┼─────────┐       ┌───────┼────────┐
 ▼        ▼         ▼       ▼       ▼        ▼
SVM       RF        LR      MLP    CNN      LSTM
KNN       GB        DT     TabNet
LightGBM AdaBoost
          │
          └────────────┬────────────┘
                       ▼
                Ensemble Model
                  SVM + RF
                       │
                       ▼
                Model Evaluation
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Accuracy     Macro F1    Weighted F1
                       │
                       ▼
               Visual Analysis
                       │
              ┌────────┴────────┐
              ▼                 ▼
       Performance Plot    Confusion Matrices
```

---

## 5. Machine Learning Models

Several traditional machine learning algorithms were implemented and evaluated.

### Support Vector Machine

A **Tuned Support Vector Machine (SVM)** was developed to classify arrhythmia observations using the preprocessed ECG features.

### Random Forest

A **Random Forest Classifier** was implemented as an ensemble of decision trees to perform multi-class arrhythmia classification.

### Logistic Regression

Logistic Regression was included as a traditional statistical machine learning classification approach.

### K-Nearest Neighbors

The **K-Nearest Neighbors (KNN)** algorithm was also evaluated for classification.

### LightGBM

**LightGBM** was implemented as a gradient-boosting-based machine learning model.

### AdaBoost

An **AdaBoost Classifier** was included among the evaluated ensemble learning approaches.

### Decision Tree

A **Decision Tree Classifier** was also trained and evaluated.

---

## 6. Deep Learning Models

The project also evaluated several neural-network-based architectures.

### Multi-Layer Perceptron

A **Multi-Layer Perceptron (MLP)** was implemented using Keras/TensorFlow.

### TabNet

**TabNet** was evaluated as a deep learning architecture designed for tabular data.

### Convolutional Neural Network

A **Convolutional Neural Network (CNN)** was implemented after reshaping the processed data to meet the model's input requirements.

### Long Short-Term Memory

A **Long Short-Term Memory (LSTM)** network was also implemented and evaluated.

---

## 7. Ensemble Model

An ensemble model was developed using a **VotingClassifier** combining:

```text
Tuned SVM
    +
Random Forest
    ↓
VotingClassifier Ensemble
```

The purpose of the ensemble was to combine the predictions of two strong classifiers and evaluate whether this combination could improve classification performance.

---

## 8. Evaluation Metrics

The performance of each model was evaluated using:

* **Accuracy**
* **Macro Average F1-Score**
* **Weighted Average F1-Score**

These metrics were used to compare the classification performance of the different machine learning and deep learning approaches.

---

## 9. Model Performance

The reported results for the leading models were:

| Model               |   Accuracy |   Macro F1 | Weighted F1 |
| ------------------- | ---------: | ---------: | ----------: |
| Ensemble (SVM + RF) | **0.9984** | **0.9984** |  **0.9984** |
| Random Forest       |     0.9953 |     0.9953 |      0.9953 |
| Tuned MLP           |     0.9937 |     0.9936 |      0.9936 |
| Tuned SVM           |     0.9890 |     0.9887 |      0.9887 |

The **SVM + Random Forest ensemble** achieved the highest reported accuracy, macro F1-score, and weighted F1-score among the models shown in the project report.

---

## 10. Model Comparison

The experimental results demonstrate strong performance across several of the evaluated models.

The reported results show:

* The **SVM + Random Forest ensemble** achieved an accuracy of **99.84%**.
* **Random Forest** achieved an accuracy of **99.53%**.
* The **Tuned MLP** achieved an accuracy of **99.37%**.
* The **Tuned SVM** achieved an accuracy of **98.90%**.

The ensemble also achieved a macro F1-score and weighted F1-score of **0.9984**.

---

## 11. Visual Analysis

The project generated several visualizations to analyze and compare model performance.

### Model Performance Comparison

A bar plot was generated to compare:

* Accuracy
* Macro Average F1-score
* Weighted Average F1-score

across the trained models.

### Confusion Matrices

Confusion matrices were generated for the **top four performing models**.

These matrices provide a detailed breakdown of correct and incorrect classifications for each arrhythmia class.

High values along the diagonal represent correctly classified observations, while off-diagonal values represent misclassifications.

---

## 12. Results and Analysis

The results indicate that ensemble and tree-based approaches performed strongly on the classification task.

The **SVM + Random Forest ensemble** produced the highest reported performance, followed by Random Forest and the Tuned MLP.

The confusion matrices provide additional information about how the models performed across individual arrhythmia classes and where classification errors occurred.

---

## 13. Project Outputs

The project produced several outputs for evaluating and visualizing model performance.

```text
Project Outputs
│
├── Model Performance Summary
│
├── Model Performance Comparison Plot
│
├── Confusion Matrix - Ensemble (SVM + RF)
│
├── Confusion Matrices - Other Top Models
│
└── Trained Model Results
```

The project report specifically references:

```text
model_performance_summary.csv
model_performance_comparison.png
Confusion_Matrix_for_Ensemble_(SVM+RF).png
```

---

## 14. Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TensorFlow
* Keras
* TabNet
* LightGBM
* Matplotlib
* SMOTE
* Principal Component Analysis (PCA)

---

## 15. Project Structure

```text
Arrhythmia-Classification/
│
├── arrhythmia_classification.ipynb
├── report.pdf
├── model_performance_summary.csv
├── model_performance_comparison.png
├── Confusion_Matrix_for_Ensemble_(SVM+RF).png
│
└── README.md
```

> **Note:** If the original `arrhythmia.csv` dataset is subject to redistribution restrictions, it should not be uploaded directly to GitHub. Instead, provide the original dataset source and instructions for obtaining it.

---

## 16. Conclusion

This project successfully applied a range of **machine learning and deep learning techniques to ECG-based arrhythmia classification**.

The models included traditional machine learning algorithms, neural-network architectures, and an ensemble model combining a tuned SVM with Random Forest.

Among the reported results, the **SVM + Random Forest ensemble achieved the highest performance**, with:

```text
Accuracy          : 0.9984
Macro F1-Score    : 0.9984
Weighted F1-Score : 0.9984
```

Random Forest and Tuned MLP also demonstrated strong classification performance.

The project provides a foundation for further development through more advanced deep learning architectures, more extensive hyperparameter optimization, and additional feature-engineering techniques.

---

## 17. Research Relevance

This project demonstrates practical experience in:

* **Healthcare and biomedical machine learning**
* **ECG signal-based classification**
* **Multi-class classification**
* **Data preprocessing**
* **Missing-value imputation**
* **Feature scaling**
* **Dimensionality reduction**
* **Class imbalance handling**
* **Ensemble learning**
* **Traditional machine learning**
* **Deep learning**
* **Model evaluation**
* **Confusion matrix analysis**
* **Machine learning visualization**

The project provides a foundation for further research in **AI-assisted cardiac health monitoring, ECG-based diagnosis, biomedical signal analysis, healthcare machine learning, and interpretable clinical prediction systems**.

