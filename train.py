from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from joblib import dump
import pandas as pd


df=pd.read_csv('data/train.csv')

X=df.drop(columns=['next_operating_profit'])
y=df['next_operating_profit']

X_train,X_test,y_train,y_test=train_test_split(
X,y,test_size=.2,random_state=42)

model=LGBMRegressor(
n_estimators=500,
learning_rate=.03,
max_depth=6
)

model.fit(X_train,y_train)

pred=model.predict(X_test)

print('MAE=',mean_absolute_error(y_test,pred))

dump(model,'models/model.pkl')