from fastapi import FastAPI, HTTPException
from .schemas import (
    TrainV2Response,
    NewProjectPredictRequest,
    PredictV2Response,
    AhspPrediction,
)
from .ml_v2.trainer import train_v2_from_db
from .ml_v2.predictor import predict_for_new_project

app = FastAPI(title="RancangHub RAB AI V2")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/train/v2", response_model=TrainV2Response)
def train_v2():
    try:
        meta = train_v2_from_db()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return TrainV2Response(
        model_version=meta["model_version"],
        mae_ensemble=meta["metrics"]["mae_ensemble"],
        mape_ensemble=meta["metrics"]["mape_ensemble"],
    )

@app.post("/predict/v2", response_model=PredictV2Response)
def predict_v2(req: NewProjectPredictRequest):
    try:
        version, df_out = predict_for_new_project(
            regency_id=req.regency_id,
            price_period_id=req.price_period_id,
            project_name=req.project_name,
            description=req.description,
            volume_threshold=0.0,  # bisa diatur
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    items = [
        AhspPrediction(
            ahsp_item_id=row["ahsp_item_id"],
            ahsp_code=row["ahsp_code"],
            ahsp_name=row["ahsp_name"],
            unit=row["ahsp_unit"],
            volume_pred=float(row["volume_pred"]),
        )
        for _, row in df_out.iterrows()
    ]

    return PredictV2Response(
        model_version=version,
        items=items,
    )
