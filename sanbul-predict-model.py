import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import tensorflow as tf
import keras
from pandas.plotting import scatter_matrix


# 1-1 데이터 로드
fires = pd.read_csv("sanbul2district-divby100.csv", sep=",")
# fires['burned_area'] = np.log(fires["burned_area"] + 1)


# 1-2 데이터 출력
print("\nfires.head():\n", fires.head())
print("\n\nfires.info():")
print(fires.info())
print("\n\nfires.describe():\n", fires.describe())
print("\n\nmonth value_counts():\n", fires['month'].value_counts())
print("\n\nday value_counts():\n", fires['day'].value_counts())


# 1-3, 1-4 데이터 시각화
fires.hist(bins=50, figsize=(20, 15))
plt.show()


# 1-5 데이터 셋 생성
train_set, test_set = train_test_split(fires, test_size=0.2, random_state=42)
test_set.head()
fires['month'].hist()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(fires, fires['month']):
    strat_train_set = fires.loc[train_idx]
    strat_test_set = fires.loc[test_idx]

print("\nMonth category proportion: \n",
      strat_test_set['month'].value_counts()/len(strat_test_set))
print("\nOverall month category proportion: \n",
      fires['month'].value_counts()/len(fires))


# 1-6 매트릭스 출력
attributes = ['burned_area', 'max_temp', 'avg_temp', 'max_wind_speed']
scatter_matrix(fires[attributes], figsize=(12, 8))
plt.show()


# 1-7 지역별 'burned_area' 플롯
fires.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
           s=fires['max_temp'], label='max_temp',
           c='burned_area', cmap=plt.get_cmap("jet"), colorbar=True)
plt.show()


# 1-8 month, day 인코딩
fires = strat_train_set.drop('burned_area', axis=1)
fires_labels = strat_train_set['burned_area'].copy()

fires_cat_month = fires[['month']]
fires_cat_day = fires[['day']]

cat_month_encoder = OneHotEncoder()
cat_day_encoder = OneHotEncoder()

fires_cat_month_encoded = cat_month_encoder.fit_transform(fires_cat_month)
fires_cat_day_encoded = cat_day_encoder.fit_transform(fires_cat_day)

print("\nfires_cat_month_encoded:\n")
print(fires_cat_month_encoded[:10])
print("\ncat_month_encoder.categories_:\n")
print(cat_month_encoder.categories_)

print("\nfires_cat_day_encoded:\n")
print(fires_cat_day_encoded[:10])
print("\ncat_day_encoder.categories_:\n")
print(cat_day_encoder.categories_)

fires_num = fires.drop(['month', 'day'], axis=1)


# 1-9 인코딩 training 셋 생성
print("\n\n#############################################################")
print("Now let's build a pipeline for preprocessing the numerical attributes:")

num_pipeline = Pipeline([
    ('std_scaler', StandardScaler())
])

fires_num_tr = num_pipeline.fit_transform(fires_num)

num_attribs = list(fires_num)
cat_attribs = ["month", "day"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)
])

fires_prepared = full_pipeline.fit_transform(fires)


# 2 Keras 모델 개발
x_train, x_valid, y_train, y_valid = train_test_split(fires_prepared, fires_labels,
                                                      test_size=0.2, random_state=42)

fires_test = strat_test_set.drop('burned_area', axis=1)
fires_test_labels = strat_test_set['burned_area'].copy()
fires_test_prepared = full_pipeline.transform(fires_test)
x_test, y_test = fires_test_prepared, fires_test_labels

np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=x_train.shape[1:]),
    keras.layers.Dense(30, activation='relu', input_shape=x_train.shape[1:]),
    keras.layers.Dense(30, activation='relu', input_shape=x_train.shape[1:]),
    keras.layers.Dense(1)
])

model.summary()

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(learning_rate=1e-3))
history = model.fit(x_train, y_train, epochs=200, validation_data=(x_valid, y_valid))

model.save('fires_model.keras')

x_new = x_test[:3]
print("\nnp.round(model.predict(x_new), 1):\n", np.round(model.predict(x_new), 2))