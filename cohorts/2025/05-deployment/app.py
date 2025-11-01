from fastapi import FastAPI
import pickle

app = FastAPI()

with open("pipeline_v1.bin", "rb") as f_in:
    model = pickle.load(f_in)

@app.post("/predict")
def predict(client: dict):
    prob = model.predict_proba([client])[0, 1]
    return {"conversion_probability": prob}
