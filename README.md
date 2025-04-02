# 🏀 SlamDunkInsights
<img src="https://github.com/user-attachments/assets/229757db-0bde-485d-8192-708eadedea05" alt="slamdunkinsights" width="500">

## 📌 Project Overview
SlamDunkInsights is a machine learning project aimed at analyzing NBA data to extract meaningful insights. The project leverages machine learning techniques such as clustering, regression, and classification to study player performance, team efficiency, and game outcomes.

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/SlamDunkInsights.git
cd SlamDunkInsights
```

### 2️⃣ Install Dependencies
Ensure you have Python 3.8+ installed, then install dependencies:
```bash
pip install -r requirements.txt
```
Or install necessary libraries directly:
```python
!pip install plotly xgboost scikit-learn
```

### 3️⃣ Run Jupyter Notebook
If you want to explore the project in an interactive environment:
```bash
jupyter notebook SlamDunkInsights.ipynb
```

## 📊 Data Preprocessing
### Standardization
Feature scaling is performed using StandardScaler to normalize the dataset:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Principal Component Analysis (PCA)
PCA is applied to reduce dimensionality while preserving variance:
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

## 📈 Clustering Analysis
### Finding Optimal K (Elbow Method & Silhouette Analysis)
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

inertia = []
silhouette_scores = []
k_values = range(2, 10)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_pca)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_pca, cluster_labels))

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(k_values, inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")

plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette_scores, marker='o')
plt.title("Silhouette Score")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.show()
```
<img src="https://github.com/user-attachments/assets/4ca5bddc-6ca5-46f0-9817-8f7228dd0684" width="800">
<img src="https://github.com/user-attachments/assets/666a67b3-f323-4b6f-82c2-a8f05e86d70a" width="800">


### Applying K-Means Clustering
```python
optimal_k = 3  # Based on elbow method and silhouette analysis
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_pca)
```
![{57BE43EE-1CCE-42E1-A443-7BEEFC9A94AC}](https://github.com/user-attachments/assets/8b685b69-17bc-411a-8ec8-68e7667e35d3)

## 🎯 Regression Model - Lasso Regression
### Implementing Lasso Regression with K-Fold Cross-Validation
```python
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

kf = KFold(n_splits=5, shuffle=True, random_state=42)
lasso = Lasso(alpha=0.1)

mse_scores = -cross_val_score(lasso, X_scaled, y, cv=kf, scoring='neg_mean_squared_error')
print(f"Mean Squared Error: {mse_scores.mean():.4f}")
```
![image](https://github.com/user-attachments/assets/3743c290-da43-4b46-a44d-36367374e515)
![image](https://github.com/user-attachments/assets/38ec1605-dec6-4eb2-889b-1a7fd41a27ff)

## 🎯 Model Training - XGBoost Classifier
The dataset is also used to train an **XGBoost** model for classification tasks.

### Splitting the Data
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

### Training the XGBoost Model
```python
import xgboost as xgb

model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
```

### Making Predictions
```python
predictions = model.predict(X_test)
```

### Evaluating the Model
```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print(f'Model Accuracy: {accuracy * 100:.2f}%')
```


