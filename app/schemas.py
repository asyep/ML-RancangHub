from pydantic import BaseModel
from typing import List

class TrainV2Response(BaseModel):
    model_version: str
    mae_ensemble: float
    mape_ensemble: float

class NewProjectPredictRequest(BaseModel):
    regency_id: str
    price_period_id: str
    project_name: str
    description: str

class AhspPrediction(BaseModel):
    ahsp_item_id: str
    ahsp_code: str
    ahsp_name: str
    unit: str
    volume_pred: float

class PredictV2Response(BaseModel):
    model_version: str
    items: List[AhspPrediction]
