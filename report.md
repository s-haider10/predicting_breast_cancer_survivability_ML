# Report: Predicting the Survivability Rate of Breast Cancer Patients

## 1. Introduction

Breast cancer is a significant public health concern worldwide, affecting millions of individuals and families each year. It is imperative to accurately predict the survivability rate of breast cancer patients to facilitate timely interventions, personalized treatment plans, and improved patient outcomes. In this project, we aimed to test various machine learning models capable of predicting the survival status of breast cancer patients based on a range of clinical and demographic features.

## 2. Data Acquisition and Preprocessing

The dataset used in this project was sourced from the NYU Big Data Science Assignment 5 and comprised 4024 instances with 16 features. The initial phase of the project involved meticulous data preprocessing steps:

- **Missing Values Handling**: We meticulously checked for missing values, employing methods to identify and handle any null or missing entries. Fortunately, no missing values were detected in the dataset.
- **Categorical Variable Encoding**: Categorical variables were encoded into numerical format using appropriate mapping techniques. This conversion enabled the models to interpret categorical data effectively during training and prediction phases.
- **Outlier Detection and Removal**: Outliers, which could potentially skew model predictions, were identified using robust statistical methods such as K Nearest Neighbors (KNN) with Euclidean distance. Outliers were then carefully removed to ensure the integrity and reliability of the dataset.

## 3. Normalization and Dimension Reduction

Normalization of the data was carried out to standardize the range of features, mitigating the impact of varying scales on model performance we used the min max formaula i.e $X_n = \frac{x-max_x}{max_x-min_x}$. Furthermore, to enhance model efficiency and reduce computational complexity, dimension reduction techniques were applied:

- **Pearson Correlation Analysis**: Highly correlated features were identified and eliminated to alleviate multicollinearity issues, thereby enhancing the robustness and interpretability of the models.

- Correlated Pairs and their coefficient
  | Feature 1 | Feature 2 | Correlation Coefficient |
  |--------------------|------------------------|-------------------------|
  | T Stage | Tumor Size | 0.8103 |
  | N Stage | 6th Stage | 0.8816 |
  | N Stage | Regional Node Positive | 0.8409 |
  | Differentiate | Grade | -1.0 |

  Hence we dropped these features: `'Grade', '6th Stage', 'Tumor Size', 'Reginol Node Positive'`

- **Sequential Forward Selection (SFS)**: Leveraging SFS, we systematically evaluated and selected the most informative features based on their entropies, thereby optimizing model performance and reducing overfitting risks

  Selected features: `['Age', 'T Stage', 'N Stage', 'differentiate', 'Estrogen Status', 'Progesterone Status', 'Regional Node Examined', 'Survival Months']`

## 4. Predictive Modeling and Evaluation

A diverse array of machine learning algorithms were employed to build predictive models, each offering unique strengths and capabilities. The performance of these models was rigorously evaluated using a suite of metrics including accuracy, precision, recall, and F1 score:

- **K Nearest Neighbors (KNN)**: A KNN algorithm was implemented, achieving a commendable accuracy of 89.80%. KNN's ability to classify instances based on similarity to neighboring data points made it a robust choice for this task.

  - Summary: KNN is a simple, instance-based learning algorithm that classifies a data point based on the majority class of its nearest neighbors.
  - Pros: Easy to understand and implement, no training phase, works well with small datasets and non-linear relationships.
  - Cons: Computationally expensive during testing phase, sensitive to irrelevant features and the choice of distance metric.
  - Main Hyperparameters:
    - `k`: Number of nearest neighbors to consider.
    - Distance metric (e.g., Euclidean distance, Manhattan distance).

- **Naïve Bayes**: Gaussian Naïve Bayes, known for its simplicity and efficiency, yielded an accuracy of 84.83%. Despite its simplistic assumptions, Naïve Bayes performed admirably, showcasing its versatility in classification tasks.

  - Summary: Naïve Bayes is a probabilistic classifier based on Bayes' theorem with the assumption of independence between features.
  - Pros: Fast training and prediction, performs well with small datasets and high-dimensional feature spaces, handles missing values well.
  - Cons: Strong independence assumption may lead to suboptimal performance in some cases, especially when features are correlated.
  - Main Hyperparameters: None (though some variants may have smoothing parameters).

- **C4.5 Decision Tree**: The decision tree algorithm, a stalwart in machine learning, delivered an accuracy of 84.45%. Decision trees' intuitive decision-making process and interpretability made them valuable assets in understanding the underlying patterns in the dataset.

  - Summary: C4.5 is a decision tree algorithm that recursively splits the data based on the feature that provides the most information gain.
  - Pros: Easy to interpret and visualize, handles both numerical and categorical data, automatically handles feature selection.
  - Cons: Prone to overfitting, sensitive to noisy data and outliers, may create biased trees for imbalanced datasets.
  - Main Hyperparameters:
    - Maximum tree depth.
    - Minimum number of samples required to split a node.
    - Minimum impurity decrease required for a split.

- **Random Forest and Gradient Boosting**: Leveraging ensemble learning techniques, Random Forest and Gradient Boosting models achieved accuracies of 89.68% and 89.55% respectively. Their ability to mitigate overfitting and handle complex relationships in the data made them formidable contenders in predictive modeling.

  - Summary: Random Forest is an ensemble learning method that constructs multiple decision trees and combines their predictions through averaging or voting.
  - Pros: Reduces overfitting compared to individual decision trees, handles high-dimensional data well, robust to noisy data.
  - Cons: More complex than single decision trees, longer training time and higher memory usage, less interpretable.
  - Main Hyperparameters:

    - Number of trees in the forest.
    - Maximum tree depth.
    - Number of features to consider at each split.
    - Minimum number of samples required to split a node.

  - Summary: Gradient Boosting builds an ensemble of weak learners (typically decision trees) sequentially, where each new model corrects errors made by the previous ones.

  - Pros: Often produces highly accurate models, handles both numerical and categorical data, less prone to overfitting compared to Random Forest.
  - Cons: More sensitive to hyperparameters and prone to overfitting with large datasets, longer training time, and higher computational cost.
  - Main Hyperparameters:
    - Learning rate (shrinkage parameter).
    - Number of trees (boosting iterations).
    - Maximum tree depth.
    - Minimum number of samples required to split a node.

## 5. Hyperparameter Tuning and Optimization

To further refine model performance and fine-tune their parameters, comprehensive hyperparameter tuning was conducted using GridSearchCV:

- **Random Forest Optimization**: Through GridSearchCV, the Random Forest classifier was optimized with 100 estimators, a maximum depth of 7, and a minimum samples split of 6, resulting in enhanced predictive capabilities.
- **Gradient Boosting Optimization**: Similarly, Gradient Boosting parameters were fine-tuned to achieve optimal performance, with a learning rate of 0.1, 50 estimators, maximum depth of 3, and minimum samples split of 2, further improving predictive accuracy.

## Analysis

| **Classifier**    | **Hyperparameters**                                                   | **Accuracy** | **Precision** | **Recall** | **F1 Score** |
| ----------------- | --------------------------------------------------------------------- | ------------ | ------------- | ---------- | ------------ |
| Random Forest     | n_estimators=100, max_depth=7, min_samples_split=6                    | 0.9073       | 0.9107        | 0.9810     | 0.9445       |
| Gradient Boosting | learning_rate: 0.1, n_estimators=50, max_depth=3, min_samples_split=2 | 0.9067       | 0.9139        | 0.9752     | 0.9436       |

Both models are fairly good performance:

- Random Forest Classifier has a slightly higher recall and F1 score
- Gradient Boosting Classifier has a slightly higher precision

## 6. Analysis and Conclusion

The comprehensive analysis and evaluation of various machine learning models underscored the efficacy of predictive modeling techniques in assessing the survivability rate of breast cancer patients. Both Random Forest and Gradient Boosting classifiers demonstrated robust performance, with Random Forest exhibiting slightly higher recall and F1 score, while Gradient Boosting achieved marginally higher precision. These models represent valuable tools in clinical decision-making, aiding healthcare practitioners in prognostic assessments and treatment planning for breast cancer patients.

## 7. Future Directions

- **Ensemble Model Integration**: Exploring the possibility of combining predictions from multiple models through ensemble techniques such as stacking or blending to further enhance predictive accuracy.
- **Feature Engineering**: Continuously refining feature engineering processes to identify novel predictors and improve model interpretability and generalizability.
- **Real-time Deployment**: Transitioning the developed models into real-world clinical settings, enabling real-time prediction and decision support for healthcare professionals.

## 8. References

- Bellaachia, Abdelghani & Guven, Erhan. (2006). Predicting Breast Cancer Survivability using Data Mining Techniques. Age. 58.
- Predictive Analytics for Dummies by Prof Anasse Bari

By employing machine learning techniques and rigorous evaluation methodologies, we have demonstrated the potential of data-driven approaches in improving patient care and clinical outcomes in the fight against breast cancer.
