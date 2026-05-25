from joblib import load
import pandas as pd

model=load('models/model.pkl')


def predict(data):

    x=pd.DataFrame([data])

    return float(model.predict(x)[0])