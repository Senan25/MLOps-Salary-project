from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from joblib import load
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import pandas as pd
from load_model import load_latest_model

# Load the model
model = load("rf_model.pkl")
#model = load_latest_model()

app = FastAPI()

# Middleware for CORS (if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files for CSS
app.mount("/static", StaticFiles(directory="static"), name="static")

# HTML Template rendering
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, age: int = Form(...), experience: int = Form(...), marital_status: str = Form(...)):
    # Transform input into the required format
    input_data = {
        "Marital Status": marital_status,
        "Experience": experience,        
        "Age": age                  
    }
    input_data = pd.DataFrame([input_data])
    prediction = model.predict(input_data)
    return templates.TemplateResponse("result.html", {"request": request, "salary": prediction[0]})

# Directory structure should include a templates folder for HTML and static for CSS.
