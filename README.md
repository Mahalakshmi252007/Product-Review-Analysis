# **🛍️ Product Review Sentiment Analysis**

**📌 Project Overview**

This project focuses on analyzing customer product reviews and automatically classifying them into three sentiment categories:

😊 Positive
😐 Neutral
😞 Negative

The project uses Transformer-based sentence embeddings to convert unstructured review text into meaningful numerical representations. Machine learning classifiers are then trained on these embeddings to predict the sentiment of new reviews.

**🎯 Problem Statement**

E-commerce platforms receive a large number of customer reviews every day. Manually analyzing these reviews is time-consuming and difficult to scale.

This project aims to build an automated sentiment analysis system that can classify product reviews and help businesses understand customer opinions more efficiently.

**🎯 Objectives**

Convert customer review text into numerical representations using Transformer embeddings.
Extract semantic meaning from product reviews.
Train and compare Random Forest and Gradient Boosting classifiers.
Evaluate model performance using Accuracy and Weighted F1-score.
Handle the challenges caused by an imbalanced sentiment dataset.
Predict sentiment for new and unseen product reviews.
Provide useful insights that can support business decision-making.

**📊 Dataset**

The dataset contains customer product reviews with the following columns:

Column	Description
Product ID	Unique identifier for the product
Product Review	Customer-written product review
Sentiment	Sentiment label: Positive, Neutral, or Negative
Dataset Statistics
Original records: 1,007
Columns: 3
Duplicate records: 2
Records after cleaning: 1,005
Missing values: 0
Sentiment classes: Positive, Neutral, Negative

The dataset is highly imbalanced, with approximately 85% Positive reviews, while Neutral and Negative reviews represent smaller portions of the dataset.

**🔄 Project Workflow**

Customer Reviews
       ↓
Data Loading
       ↓
Data Cleaning
       ↓
EDA
       ↓
Transformer Sentence Embeddings
       ↓
Train/Test Split
       ↓
Model Training
   ↙           ↘
Random Forest   Gradient Boosting
   ↓               ↓
Model Evaluation
       ↓
Model Comparison
       ↓
Final Prediction
       ↓
Sentiment + Confidence Score

**🧹 Data Preprocessing**

The following preprocessing steps were performed:

Loaded the product review dataset.
Checked the dataset shape and structure.
Checked for missing values.
Identified duplicate records.
Removed 2 duplicate rows.
Reset the DataFrame index.
Analyzed the sentiment distribution.

After preprocessing, the dataset contained 1,005 unique reviews.

**🔍 Exploratory Data Analysis**

EDA was performed to understand the distribution of sentiment classes.

The analysis showed that the dataset is highly imbalanced, with Positive reviews forming the majority class.

Because of this imbalance, model evaluation was performed using Weighted F1-score along with Accuracy, rather than relying only on accuracy.

**🤖 Transformer-Based Embeddings**

The project uses the pre-trained:

sentence-transformers/all-MiniLM-L6-v2

This Transformer model converts each product review into a dense numerical vector that captures the semantic meaning of the text.

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    'sentence-transformers/all-MiniLM-L6-v2'
)

The generated embeddings are then used as input features for the machine learning classifiers.

🌲 Machine Learning Models

Two supervised machine learning models were trained using the Transformer embeddings:

1. Random Forest

Random Forest was trained to classify reviews into:

Positive
Neutral
Negative
2. Gradient Boosting

Gradient Boosting was also trained using the same Transformer-generated embeddings.

The two models were evaluated using the same test dataset for a fair comparison.

**📈 Model Performance**

Model	Test Accuracy	Weighted F1-Score
Random Forest + Transformer	86.6%	81.8%
Gradient Boosting + Transformer	84.1%	80.3%

The Random Forest model was selected as the final model based on its test-set performance and comparatively better generalization.

Note: The Random Forest training accuracy was 100%, while test accuracy was 86.6%, indicating some overfitting.

**🔮 Prediction on New Reviews**

The final model can process completely new reviews through the following pipeline:

New Review
    ↓
Sentence Transformer
    ↓
Embedding
    ↓
Random Forest
    ↓
Predicted Sentiment
    ↓
Confidence Score

This demonstrates an end-to-end inference workflow for unseen product reviews.

**💼 Business Applications**

The sentiment analysis system can help businesses:

📦 Understand customer satisfaction.
🔍 Identify negative product feedback.
🚨 Detect potential product issues.
💡 Discover insights from neutral reviews.
📈 Monitor changes in customer sentiment.
🎯 Support product improvement decisions.
💬 Improve customer support prioritization.
📊 Analyze large volumes of reviews automatically.

**🛠️ Technologies Used**

Python
Pandas
NumPy
Scikit-learn
Sentence Transformers
Hugging Face Transformers
PyTorch
Matplotlib
Seaborn
Google Colab

**📚 Key Concepts**

This project demonstrates practical knowledge of:

Natural Language Processing (NLP)
Sentiment Analysis
Exploratory Data Analysis
Data Cleaning
Transformer Models
Sentence Embeddings
Supervised Machine Learning
Random Forest
Gradient Boosting
Multi-class Classification
Imbalanced Dataset Handling
Model Evaluation
Model Generalization
Text Classification

**📌 Results & Conclusion**

The project successfully demonstrates an end-to-end Product Review Sentiment Analysis pipeline using Transformer-based sentence embeddings and machine learning classifiers.

The Random Forest + Transformer model achieved 86.6% test accuracy and 81.8% weighted F1-score, while Gradient Boosting achieved 84.1% test accuracy and 80.3% weighted F1-score.

The project also demonstrates that Transformer embeddings can be combined with traditional machine learning algorithms to perform semantic text classification and predict sentiment for previously unseen customer reviews.
