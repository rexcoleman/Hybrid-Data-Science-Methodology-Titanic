# A Framework for Solving Data Science Problems

## Executive Summary

Welcome to my Framework for Solving Data Science Problems. This repository provides a comprehensive framework for solving data science problems. The project builds upon one of the most popular [Kaggle notebooks](https://www.kaggle.com/code/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy), leveraging best-in-class methodologies to create a reliable foundation for solving data science problems. By reproducing and substantially building upon this work, I aim to illustrate the value of learning from top practitioners while also solving one of the most important problems in data science.

Rushing into a data science project without a structured approach can lead to numerous problems, which can severely impact project success, cost, and outcomes. This project addresses these issues by implementing a well-defined framework and best practices ensuring thorough problem understanding, effective data preprocessing, and robust model evaluation.

Borrowing fron the giants of Agile, DevOps, and Lean Entrepreneurship, we are leveragine the concept of ['shifting left'](https://en.wikipedia.org/wiki/Shift-left_testing) to support a more flexible and adaptive development process, facilitating faster delivery of high-quality data science solutions that align with business needs.

This project applies the above concepts to the popular "Titanic - Machine Learning from Disaster" Kaggle competition and can applied generally to a wide array of data science problems.

Thank you for visiting my repository. I hope this project inspires you to implement a structured approach to avoid common data science pitfalls.  I welcome comments: especially those that will help improve upon this concept.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Introduction](#introduction)
3. [Data Science Framework](#data-science-framework)
4. [Project Details](#project-details)
5. [Reproducing Best-in-Class Work](#reproducing-best-in-class-work)
6. [Addressing the Objection of Copying Work](#addressing-the-objection-of-copying-work)
7. [Technologies Used](#technologies-used)
8. [Getting Started](#getting-started)
9. [Results and Insights](#results-and-insights)
10. [Contributing](#contributing)
11. [Contact](#contact)

---

## Introduction

The motivation behind this project is twofold:

1. **Learning from the Best**: By reproducing work from top data scientists, we can gain valuable insights and understand the methodologies that lead to successful projects.
2. **Framework Development**: Creating a robust framework for data science projects that can be applied to various datasets and problems, ensuring a structured approach to avoid common pitfalls.

### Learning from the Best

Why reinvent the wheel when you don't have to.  To mix metaphors, let's stand on the shoulders of giants and improve upon their work!


### Problems Related to Rushing into a Data Science Project

Rushing into a data science project without a structured approach can lead to numerous problems, severely impacting the success, reliability, and cost of the outcomes. Below is an exhaustive list of these potential issues:

1. **Inadequate Problem Understanding**:
   - Misalignment with business objectives.
   - Unclear problem definition leading to irrelevant solutions.

2. **Poor Data Collection and Exploration**:
   - Missing critical data points.
   - Overlooking data quality issues.
   - Failure to understand the data distribution and patterns.

3. **Insufficient Data Cleaning and Preprocessing**:
   - Presence of noisy or irrelevant data.
   - Incorrect handling of missing values.
   - Inconsistent data formatting and scaling.

4. **Ineffective Feature Engineering**:
   - Missing out on key features that improve model performance.
   - Overfitting due to too many features.
   - Ignoring domain knowledge in feature selection.

5. **Inappropriate Model Selection and Training**:
   - Choosing models that are not suitable for the problem.
   - Not validating the model selection process.
   - Inadequate training leading to underfitting or overfitting.

6. **Lack of Proper Model Evaluation and Validation**:
   - Using incorrect metrics for model evaluation.
   - Not performing cross-validation to ensure model generalization.
   - Ignoring potential data leakage during validation.

7. **Inadequate Hyperparameter Tuning**:
   - Suboptimal model performance due to default hyperparameters.
   - Time-consuming trial and error without a systematic approach.

8. **Misinterpretation of Results**:
   - Drawing incorrect conclusions from model outputs.
   - Failure to consider model limitations and biases.

9. **Poor Model Deployment (if applicable)**:
   - Incompatibility with production environment.
   - Lack of monitoring and maintenance plan.

10. **Inadequate Documentation and Reporting**:
    - Difficult for others to understand and reproduce the work.
    - Lack of transparency in methodologies and results.

By addressing these issues through a structured approach, as demonstrated in this project, we can significantly improve the quality and reliability of data science outcomes.

### Benefits of Shifting Left in Data Science Projects:

1. **Enhanced Quality and Reliability**: Early identification and resolution of issues improve the overall reliability of the project.
2. **Cost and Time Efficiency**: Addressing issues early reduces the cost and time associated with fixing problems later.
3. **Better Stakeholder Engagement**: Early involvement of stakeholders ensures the project remains aligned with business goals.
4. **Structured Approach**: A methodical approach from the beginning ensures a systematic way to tackle data science projects.

This framework serves as a guide for tackling data science projects methodically and effectively.


---

## Data Science Framework

1. ### Define the Problem
   If data science, big data, machine learning, predictive analytics, business intelligence, or any other buzzword is the solution, then what is the problem? Problems should precede requirements, requirements should precede solutions, solutions should precede design, and design should precede technology.

   Kaggle clearly defined this problem for us.

3. ### Gather the Data
   Chances are, the dataset(s) already exist somewhere. It may be external or internal, structured or unstructured, static or streamed, objective or subjective. The goal is to find and consolidate these datasets.

   Kaggle provided a clean dataset.

5. ### Prepare Data for Consumption
   Data wrangling is a required process to turn “wild” data into “manageable” data. This includes data extraction, data cleaning, and preparing data for analysis by implementing data architectures, developing data governance standards, and ensuring data quality.

   Used 4 C's of Data Cleaning: Correcting, Completing, Creating, Converting.

6. ### Perform Exploratory Data Analysis
   Deploy descriptive and graphical statistics to look for potential problems, patterns, classifications, correlations, and comparisons in the dataset. Data categorization is also important to select the correct hypothesis test or data model.

7. ### Model Data
   Data modeling can either summarize the data or predict future outcomes. The dataset and expected results determine the algorithms available for use. Algorithms are tools that must be selected appropriately for the job.

   This report includes a generalized framework for model selection. This report presents a baseline back-of-the-envelope calculated decision tree and compares resultes from the following models. This report also implementes the following techniques to improve model results: hyper-parameter tuning and recursive feature elimination.

   ```python
   MLA = [
       # Ensemble Methods
       ensemble.AdaBoostClassifier(),
       ensemble.BaggingClassifier(),
       ensemble.ExtraTreesClassifier(),
       ensemble.GradientBoostingClassifier(),
       ensemble.RandomForestClassifier(),
       
       # Gaussian Processes
       gaussian_process.GaussianProcessClassifier(),
       
       # GLM
       linear_model.LogisticRegressionCV(),
       linear_model.PassiveAggressiveClassifier(),
       linear_model.RidgeClassifierCV(),
       linear_model.SGDClassifier(),
       linear_model.Perceptron(),
       
       # Naive Bayes
       naive_bayes.BernoulliNB(),
       naive_bayes.GaussianNB(),
       
       # Nearest Neighbor
       neighbors.KNeighborsClassifier(),
       
       # SVM
       svm.SVC(probability=True, max_iter=10000),
       svm.NuSVC(probability=True, max_iter=10000),
       svm.LinearSVC(max_iter=10000),
       
       # Trees    
       tree.DecisionTreeClassifier(),
       tree.ExtraTreeClassifier(),
       
       # Discriminant Analysis
       discriminant_analysis.LinearDiscriminantAnalysis(),
       discriminant_analysis.QuadraticDiscriminantAnalysis(),
       
       # xgboost
       XGBClassifier()    
   ]


8. ### Validate and Implement Data Model
Test your model to ensure it hasn't overfit or underfit your dataset. Determine if your model generalizes well by validating it with a subset of data not used during training.

Cross validation and ensembling were used improve generalization.


7. ### Optimize and Strategize
Iterate through the process to make the model better. As a data scientist, your strategy should be to focus on recommendations and design while outsourcing developer operations and application plumbing.

---

## Project Details

- **Dataset**: [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)
- **Objective**: The competition objective is simple: use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.


---

## Project Details



---

## Reproducing Best-in-Class Work

Reproducing high-quality work from leading data scientists provides several benefits:
- **Benchmarking**: Establishes a performance benchmark to compare our models against.
- **Learning**: Understand the best practices and methodologies used by top practitioners.
- **Innovation**: Builds a foundation upon which new ideas and improvements can be developed.

---

## Addressing the Objection of Copying Work

While reproducing work might seem like copying, it is important to recognize the value of this practice:
- **Educational Value**: Provides a hands-on learning experience, reinforcing theoretical knowledge.
- **Skill Enhancement**: Helps in honing practical data science skills by working on real-world problems.
- **Foundation for Innovation**: Enables building upon existing solutions to create improved or new methodologies.

---

## Technologies Used

- **Programming Languages**: Python version 3.10.13
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, SciPy, Scikit-learn, itertools, Graphviz, os, sys, IPython, random, time, XGBoost
- **Tools**: Jupyter Notebooks, Git, GitHub, Kaggle

---

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

1. **Clone the repository**:
    ```bash
    git clone git@github.com:rexcoleman/GeneralizedDataScienceFramework-Titanic.git
    ```
2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3. **Run the Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```

---

## Results and Insights

The project results include detailed analysis, model performance metrics, and visualizations that provide insights into the predictive power of the models used.


![Model results table](Images/ModelMeasurementsPlot.png)

Our top four models (test accuracy) are BaggineClassifier, BernoulliNB, XGBClassifier and EnsembleHardVoting. 

![Model results plot](Images/ModelMeasurementsTable.png)

Defining avoidable bias as 1.0 - train accuracy.  1.0 is argeuably an unrealistic figure for human level error.  We can justify it given that over 250 submissions reported perfect accuracy scores.

Defining variance as the spreaed between test error and train error.  Typically it is better to use the difference in dev errer (validation error) and training error.  In the case of our models, there is a wide margin between validation error and test error so we are including it in our variance error calculation.

![Model variance indicator](Images/BiasVariancePlot.png)

As a general rule for model performance, we want to work on improving the greater error (bias or variance).  

We can potentially improve model bias with: 
- Use a larger neural network
- Train longer
- Use better optimization algorithms, e.g. momemtum, RMSProp, Adam
- Search for better architecture/hyperparameters, e.g. RNN, CNN

We can potentially improve model variance with: 
- More data
- Regularization, e.g. LS, droppout, data augmentation
- Search for better architecture/hyperparameters, e.g. RNN, CNN


## Observations:
1. The Bagging Classifier model produced the highest Kaggle accuracy score: **0.78468**.
2. The Bagging Classifier model performed better than the ensemble model Kaggle accuracy scores: **0.77511** (hard voting), **0.76555** (soft voting).
3. The Bagging Classifier model appears to have the lowest variance compared to the other models.
4. The Bagging Classifier model underperformed when compared to the [TensorFlow Decision Forest model](https://www.kaggle.com/code/rexcoleman/titanic-w-tensorflow-decision-forest-rex-copy) with a Kaggle accuracy score of **0.80143**.

## Areas For Future Research:
1. Why does the TensorFlow Decision Forest model outperform all models in this notebook?
2. Why do several models in this notebook outperform the ensemble models? How can we improve the ensemble models?
3. How can I speed up the ensemble / hyperparameter tuning code?
4. What hyperparameters tuning can I use to improve performance?
5. How are some of the submissions achieving 100% accuracy Kaggle scores?
6) Find coorelation heat map code that doesn't require forcing Kaggle version control.

---

## Contributing

Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) for more information.

---

## Contact

If you have any questions or want to connect, feel free to reach out to me via [LinkedIn]([https://www.linkedin.com/in/yourprofile](https://www.linkedin.com/in/rexcoleman/)) or [email](rexcoleman@gmail.com).

---

Thank you for visiting my repository. I hope this project inspires you to implement a structured approach to avoid common data science pitfalls.
