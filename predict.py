import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt


# 재난 정보가 병홥된 CSV 파일 경로
file_path = r"C:\Users\김소민\Desktop\사문\빅콘\disaster_tr.csv"  
dis2= pd.read_csv(file_path) 

from xgboost import XGBRegressor

# 범주형 변수 categorical 타입으로 설정
dis = pd.get_dummies(dis, columns=['CST_NM'], drop_first=True)

# 피처와 타겟 변수 분리
X = dis.drop(columns=['IEM_CNT'])
y = np.log1p(dis['IEM_CNT'])  # 로그 변환

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
lgbm_model = LGBMRegressor(
    n_estimators=600,
    learning_rate=0.01,
    max_depth=20,
    min_gain_to_split=0.01,   # 분할을 위한 최소 이득
    min_data_in_leaf=10,      # 리프에 있어야 하는 최소 데이터 수
    random_state=42
)
# Stacking의 XGBoost에 enable_categorical=True 추가
xgb_model = XGBRegressor(n_estimators=700,
    learning_rate=0.01,
    max_depth=30,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42, enable_categorical=True)
base_models = [
    ('xgb', xgb_model),
    ('lgbm', lgbm_model),
    ('rf', RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42))
]
stacking_model = StackingRegressor(estimators=base_models, final_estimator=Ridge())

# 모델 학습 및 예측
stacking_model.fit(X_train, y_train)
y_pred_log = stacking_model.predict(X_test)
y_pred = np.expm1(y_pred_log)  # 로그 변환 해제
y_test_original = np.expm1(y_test)

# 평가
rmse = mean_squared_error(y_test_original, y_pred, squared=False)
r2 = r2_score(y_test_original, y_pred)

print("Stacking Ensemble Model with Categorical Encoding")
print(f"RMSE: {rmse}")
print(f"R^2 Score: {r2}")


# 학습 데이터셋 실제 값 vs 예측 값 선 그래프
plt.figure(figsize=(12, 6))
plt.plot(y_train_original.values, color="skyblue", label="Actual IEM_CNT")
plt.plot(y_train_pred, color="blue", label="Predicted IEM_CNT", alpha=0.7)
plt.xlabel("data index")
plt.ylabel("IEM_CNT")
plt.title("Actual vs Predicted")
plt.legend()
plt.show()
