# app.py
import pickle
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import JSONResponse

model_file = "pipeline_v1.bin"

# Load DictVectorizer and model
with open(model_file, "rb") as f_in:
    dv, model = pickle.load(f_in)

app = FastAPI(title="churn")

class Customer(BaseModel):
    # use Optional[...] = None so extra fields are allowed/missing fields default to None
    lead_source: Optional[str] = None
    number_of_courses_viewed: Optional[int] = None
    annual_income: Optional[float] = None

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predictfast")
def predict(payload: Customer):
    try:
        # Convert to plain dict for DictVectorizer (keeps None values; dv will handle/ignore depending on training)
        customer_dict: Dict[str, Any] = payload.model_dump()
        X = dv.transform([customer_dict])
        y_pred = model.predict_proba(X)[0, 1]
        churn = y_pred >= 0.5

        return JSONResponse(
            {
                "churn_probability": float(y_pred),
                "churn": bool(churn),
            }
        )
    except Exception as e:
        # Surface a clean error instead of HTML
        raise HTTPException(status_code=500, detail=str(e))
