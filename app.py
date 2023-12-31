from fastapi import FastAPI
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    '*'
]
app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials= True,
    allow_methods = ['*'],
    allow_headers  = ['*']
)

# Load the pre-trained model
model = joblib.load("FertilizerPkl.pkl")

'''
Temp Request Body:
{
    "Temperature":32,
    "Humidity":62,
    "Moisture":34,
    "Soil_Type":9,
    "Crop_Type":3,
    "Nitrogen":22,
    "Potassium":0,
    "Phosphorous":20
}

uvicorn app:app
'''

@app.get("/")
def index():
    return {"Message":"Hello world"}

@app.post("/predict")
def predict(data: dict):
    Temperature = data['Temperature']
    Humidity =    data['Humidity']
    Moisture =    data['Moisture']
    Soil_Type =   data['Soil_Type']
    Crop_Type =   data['Crop_Type']
    Nitrogen =    data['Nitrogen']
    Potassium =   data['Potassium']
    Phosphorous = data['Phosphorous']
    features = [Temperature,Humidity,Moisture,Soil_Type,Crop_Type,Nitrogen,Potassium,Phosphorous]
    prediction = model.predict([features])
    return {"prediction": prediction.tolist()}
