순서
1. Feature Based Ensemble
2. KNN
3.Logistic Regression
4.Support Vector Machine
5.Random Forest
6.Gradient Boosting
7.FCNN
8.LSTM
9.DNN
10.RNN
11.Wide & Deep
12. Random Forest with SMOTE+Tomek links 
13. Balanced Random Forest 
14. Random Forest with SMOTEENN
15. Figure1
16. Figure2


1. Feature-based-ensemble

# 1. 패키지 설치
!pip install -q imbalanced-learn

# 2. 코드 실행 (SMOTE 0.6)
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import pandas as pd
import itertools

# 데이터 불러오기 및 타겟 이진화 (Q3 기준)
data = load_diabetes()
X, y = data.data, data.target
threshold = np.percentile(y, 75)
y_binarized = (y > threshold).astype(int)
feature_names = data.feature_names

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y_binarized, test_size=0.2, random_state=42, stratify=y_binarized)

# 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE + RUS
smote = SMOTE(sampling_strategy=0.6, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

rus = RandomUnderSampler(sampling_strategy=0.66, random_state=42)
X_balanced, y_balanced = rus.fit_resample(X_resampled, y_resampled)

# 변수 중요도 기반 feature 조합
rf_temp = RandomForestClassifier(random_state=42)
rf_temp.fit(X_balanced, y_balanced)
importances = rf_temp.feature_importances_
feature_ranking = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
top_features = [name for name, _ in feature_ranking]
feature_index_map = {name: idx for idx, name in enumerate(feature_names)}

# 상위 10개 조합 학습
feature_combinations = list(itertools.combinations(top_features, 2))[:10]
base_learners = []
for i, (f1, f2) in enumerate(feature_combinations):
    idx1, idx2 = feature_index_map[f1], feature_index_map[f2]
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_balanced[:, [idx1, idx2]], y_balanced)
    base_learners.append((f"rf_{i}", clf))

# 앙상블 모델
ensemble = VotingClassifier(estimators=base_learners, voting='soft')
ensemble.fit(X_balanced[:, :], y_balanced)

# 평가
y_pred = ensemble.predict(X_test_scaled)
y_proba = ensemble.predict_proba(X_test_scaled)[:, 1]

# 결과 출력
results = {
    "Accuracy": accuracy_score(y_test, y_pred),
 
    "AUC": roc_auc_score(y_test, y_proba)
}
pd.DataFrame([results])
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='orange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Ensemble Model')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


2. K-Nearest Neighbors

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

# 1. 데이터 로드
data = load_diabetes()
X, y = data.data, data.target

# 2. Q3 기준 타겟 이진화
threshold = np.percentile(y, 75)
y_binarized = (y > threshold).astype(int)

# 3. 데이터 분할 (8:2, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binarized, test_size=0.2, random_state=42, stratify=y_binarized
)

# 4. 정규화
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. KNN 모델 정의 및 학습
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# 6. 예측
y_pred = knn_model.predict(X_test)
y_pred_prob = knn_model.predict_proba(X_test)[:, 1]

# 7. 평가
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

print(f"KNN Accuracy: {accuracy:.4f}")
print(f"KNN AUC: {auc:.4f}")

# 8. ROC Curve 시각화 (선택적)
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='navy', lw=2, label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - K-Nearest Neighbors')
plt.legend(loc='lower right')
plt.grid()
plt.show()




3. Logistic Regression
# Assuming X and y are already defined from previous cells
# Create y_binarized
y_binarized = (y > np.median(y)).astype(int)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 데이터 스케일링 (Assume X, y_binarized are defined)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binarized, test_size=0.3, random_state=42)

# 모델 학습
log_reg_model = LogisticRegression(random_state=42)
log_reg_model.fit(X_train, y_train)

# 예측 및 확률
y_pred = log_reg_model.predict(X_test)
y_pred_prob = log_reg_model.predict_proba(X_test)[:, 1]

# 정확도 및 AUC
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

print(f"Logistic Regression Accuracy: {accuracy:.4f}")
print(f"Logistic Regression AUC: {auc:.4f}")

# AUC Curve 시각화
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkred', lw=2, label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc='lower right')
plt.grid()
plt.show()

4. Support Vector Machine

from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 데이터 준비 (Assume X_scaled, y_binarized already exist)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binarized, test_size=0.3, random_state=42)

# SVM 모델 학습
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# 예측값 및 확률
y_pred_svm = svm_model.predict(X_test)
y_pred_svm_prob = svm_model.predict_proba(X_test)[:, 1]

# 정확도 및 AUC 점수 계산
accuracy_svm = accuracy_score(y_test, y_pred_svm)
auc_svm = roc_auc_score(y_test, y_pred_svm_prob)

print(f"SVM Accuracy: {accuracy_svm:.4f}")
print(f"SVM AUC: {auc_svm:.4f}")

# AUC Curve 시각화
fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_test, y_pred_svm_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr_svm, tpr_svm, color='orange', lw=2, label=f'ROC Curve (AUC = {auc_svm:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Support Vector Machine')
plt.legend(loc='lower right')
plt.grid()
plt.show()




5. Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 데이터 분할 (X_scaled, y_binarized 준비되어 있다고 가정)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binarized, test_size=0.3, random_state=42)

# 모델 학습
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# 예측 및 확률
y_pred = rf_model.predict(X_test)
y_pred_prob = rf_model.predict_proba(X_test)[:, 1]

# 정확도 및 AUC
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

print(f"Random Forest Accuracy: {accuracy:.4f}")
print(f"Random Forest AUC: {auc:.4f}")

# AUC Curve 시각화
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend(loc='lower right')
plt.grid()
plt.show()

6. Gradient Boosting
 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 데이터 분할 (X_scaled, y_binarized 준비되어 있다고 가정)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binarized, test_size=0.3, random_state=42)

# 모델 학습
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# 예측 및 확률
y_pred = gb_model.predict(X_test)
y_pred_prob = gb_model.predict_proba(X_test)[:, 1]

# 정확도 및 AUC
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

print(f"Gradient Boosting Accuracy: {accuracy:.4f}")
print(f"Gradient Boosting AUC: {auc:.4f}")

# ROC Curve 시각화
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='purple', lw=2, label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Gradient Boosting')
plt.legend(loc='lower right')
plt.grid()
plt.show()





7. FCNN

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 1. 데이터 로드 및 준비
data = load_diabetes()
X, y = data.data, data.target

# 타겟값 이진화 (중앙값 기준으로 분류)
y = (y > np.percentile(y, 75)).astype(int)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. FCNN 모델 생성 및 학습
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1)

# 3. 평가 및 AUC Curve 시각화
# 예측 확률 및 클래스
y_pred_prob = model.predict(X_test).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

# 정확도 및 AUC 점수
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)
print(f"Accuracy: {accuracy:.2f}")
print(f"AUC: {auc:.2f}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='green', lw=2, label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - FCNN Model')
plt.legend(loc='lower right')
plt.grid()
plt.show()




8. LSTM 

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 1. Load diabetes dataset
data = load_diabetes()
X, y = data.data, data.target

# Binarize the target (e.g., threshold at median)
threshold = np.percentile(y, 75)
y_binary = (y > threshold).astype(int)

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# 3. Feature scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape for LSTM input (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# 4. Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# 5. Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# 6. Evaluate the model
y_pred_prob = model.predict(X_test).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

print(f"Accuracy: {accuracy:.2f}")
print(f"AUC: {auc:.2f}")

# 7. Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - LSTM Model')
plt.legend(loc='lower right')
plt.grid()
plt.show()

9. DNN

!pip install scikit-learn tensorflow imbalanced-learn

import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import tensorflow as tf

# 1. 데이터 로드 및 전처리
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# 타겟 변수를 이진 분류로 변환 (중앙값 기준)
threshold = np.percentile(y, 75)
y_binary = (y > threshold).astype(int)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# 특성 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. DNN 모델 구축
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # sigmoid for binary classification
])

# 3. 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. 모델 학습
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# 5. 평가
y_pred_proba = model.predict(X_test).ravel()
y_pred = (y_pred_proba > 0.5).astype(int)

# 정확도 및 AUC
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"DNN Accuracy: {accuracy:.4f}")
print(f"DNN AUC: {auc:.4f}")

# AUC Curve 시각화
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='teal', lw=2, label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Deep Neural Network')
plt.legend(loc='lower right')
plt.grid()
plt.show()


10. RNN

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout

# 1. 데이터 로드 및 전처리
data = load_diabetes()
X = data.data
y = data.target

# 타겟 이진화
y = (y > np.percentile(y, 75)).astype(int)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# RNN 입력 차원 변경
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# 2. RNN 모델 정의
model = Sequential([
    SimpleRNN(32, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. 훈련
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1)

# 4. 평가
y_pred_prob = model.predict(X_test).ravel()
y_pred_class = (y_pred_prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred_class)
auc = roc_auc_score(y_test, y_pred_prob)

print(f"RNN Accuracy: {accuracy:.4f}")
print(f"RNN AUC: {auc:.4f}")

# 5. AUC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkblue', lw=2, label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - RNN Model')
plt.legend(loc='lower right')
plt.grid()
plt.show()



11. Wide & Deep Network

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate

# 1. 데이터 로드 및 전처리
data = load_diabetes()
X, y = data.data, data.target

# 타겟값 이진화
y_binary = (y > np.percentile(y, 75)).astype(int)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Wide & Deep 모델 정의
input_layer = Input(shape=(X_train_scaled.shape[1],))

# Wide part
wide_output = Dense(1, activation='linear')(input_layer)

# Deep part
deep = Dense(128, activation='relu')(input_layer)
deep = Dense(64, activation='relu')(deep)
deep = Dense(32, activation='relu')(deep)
deep_output = Dense(1, activation='linear')(deep)

# 결합 및 출력
merged = Concatenate()([wide_output, deep_output])
final_output = Dense(1, activation='sigmoid')(merged)

# 모델 생성
model = Model(inputs=input_layer, outputs=final_output)

# 3. 컴파일 및 학습
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_scaled, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1)

# 4. 평가 및 예측
y_pred_prob = model.predict(X_test_scaled).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

# 정확도 및 AUC
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

print(f"Wide & Deep Accuracy: {acc:.4f}")
print(f"Wide & Deep AUC: {auc:.4f}")

# 5. ROC Curve 시각화
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkgreen', lw=2, label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1, label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Wide & Deep Model')
plt.legend(loc='lower right')
plt.grid()
plt.show()


12.Random Forest with SMOTE+Tomek links 
# 1. 패키지 설치 (처음 실행 시만 필요)
# !pip install -q imbalanced-learn

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTETomek
import numpy as np
import matplotlib.pyplot as plt

# 데이터 로딩 및 이진화 (Q3 기준)
data = load_diabetes()
X, y = data.data, data.target
threshold = np.percentile(y, 75)
y_bin = (y > threshold).astype(int)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y_bin, test_size=0.2, random_state=42, stratify=y_bin
)

# 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE + Tomek Links 적용
smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X_train_scaled, y_train)

# 모델 학습
clf = RandomForestClassifier(random_state=42)
clf.fit(X_resampled, y_resampled)

# 예측 및 평가
y_pred = clf.predict(X_test_scaled)
y_proba = clf.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {acc:.4f}")
print(f"AUC: {auc_score:.4f}")

# ROC Curve 시각화
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='green', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest with SMOTE + Tomek Links')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


13. Balanced Random Forest
# 1. 패키지 설치 (처음 실행 시만)
# !pip install -q imbalanced-learn

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from imblearn.ensemble import BalancedRandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

# 데이터 로딩 및 이진화 (Q3 기준)
data = load_diabetes()
X, y = data.data, data.target
threshold = np.percentile(y, 75)
y_bin = (y > threshold).astype(int)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y_bin, test_size=0.2, random_state=42, stratify=y_bin
)

# 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Balanced Random Forest 모델 학습
brf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
brf.fit(X_train_scaled, y_train)

# 예측 및 평가
y_pred = brf.predict(X_test_scaled)
y_proba = brf.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {acc:.4f}")
print(f"AUC: {auc_score:.4f}")

# ROC Curve 시각화
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkblue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Balanced Random Forest')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


14. Random Forest with SMOTEENN
# 1. 패키지 설치 (처음 실행 시에만 필요)
# !pip install -q imbalanced-learn

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from imblearn.combine import SMOTEENN
import numpy as np
import matplotlib.pyplot as plt

# 데이터 로딩 및 이진화 (Q3 기준)
data = load_diabetes()
X, y = data.data, data.target
threshold = np.percentile(y, 75)
y_bin = (y > threshold).astype(int)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y_bin, test_size=0.2, random_state=42, stratify=y_bin
)

# 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTEENN 적용
smote_enn = SMOTEENN(random_state=42)
X_balanced, y_balanced = smote_enn.fit_resample(X_train_scaled, y_train)

# 모델 훈련
clf = RandomForestClassifier(random_state=42)
clf.fit(X_balanced, y_balanced)

# 예측 및 평가
y_pred = clf.predict(X_test_scaled)
y_proba = clf.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {acc:.4f}")
print(f"AUC: {auc_score:.4f}")

# ROC Curve 시각화
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest with SMOTEENN')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

15. Figure 1 코드

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 데이터 불러오기
data = load_diabetes()
X, y = data.data, data.target
feature_names = data.feature_names

# 2. 타겟 이진화 (Q3 기준)
threshold = np.percentile(y, 75)
y_binarized = (y > threshold).astype(int)

# 3. 데이터 분할 및 정규화
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binarized, test_size=0.2, random_state=42, stratify=y_binarized
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. 랜덤 포레스트 학습 (Feature Importance 추출용)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
importances = rf.feature_importances_

# 5. 시각화
sorted_idx = np.argsort(importances)
plt.figure(figsize=(10, 6))
plt.barh(range(len(importances)), importances[sorted_idx], align="center")
plt.yticks(range(len(importances)), [feature_names[i] for i in sorted_idx])
plt.xlabel("Feature Importance (MDI)")
plt.title("Figure 1. Feature Importance in Scikit-learn Diabetes Dataset")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


16. Figure3  코드
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# 데이터 불러오기 및 이진화
data = load_diabetes()
X, y = data.data, data.target
threshold = np.percentile(y, 75)
y_bin = (y > threshold).astype(int)

# 학습/테스트 분할
X_train, _, y_train, _ = train_test_split(X, y_bin, test_size=0.2, stratify=y_bin, random_state=42)

# 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 분포 확인 (Before)
count_before = Counter(y_train)

# SMOTE + RUS 적용
smote = SMOTE(sampling_strategy=0.6, random_state=42)
X_smote, y_smote = smote.fit_resample(X_train_scaled, y_train)
rus = RandomUnderSampler(sampling_strategy=0.66, random_state=42)
X_final, y_final = rus.fit_resample(X_smote, y_smote)

# 분포 확인 (After)
count_after = Counter(y_final)

# 시각화
labels = ['Majority', 'Minority']
before = [count_before[0], count_before[1]]
after = [count_after[0], count_after[1]]
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
bar1 = ax.bar(x - width/2, before, width, label='Before SMOTE and RUS', color='gray')
bar2 = ax.bar(x + width/2, after, width, label='After SMOTE and RUS', color='steelblue')

ax.set_ylabel('Number of Samples')
ax.set_title('Class Distribution Before and After SMOTE and RUS')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.6)

for bar in bar1 + bar2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 5, str(height), ha='center')

plt.tight_layout()
plt.show()

