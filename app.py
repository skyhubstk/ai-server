from fastapi import FastAPI
from predictor import predict
from gpt_service import summarize
from feature_engineering import create_features
from pydantic import BaseModel

app=FastAPI()

class Req(BaseModel):
    company:str
    trend:str
    risk:str
    revenue_0:float
    revenue_1:float
    revenue_2:float
    revenue_3:float
    revenue_4:float
    op_0:float
    op_1:float
    op_2:float
    op_3:float
    op_4:float
    debt_4:float
    equity_4:float

@app.post('/api/v1/analysis')
def analyze(req:Req):

    data=req.dict()

    features=create_features(data)

    pred=predict(features)

    summary=summarize(data,pred)

    return {
      'prediction':pred,
      'summary':summary
    }