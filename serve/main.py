from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

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

@app.post("/predict")
async def predict(age: int = Form(...), experience: int = Form(...), marital_status: str = Form(...)):
    # Transform input into the required format
    input_data = [[age, experience, marital_status]]
    prediction = model.predict(input_data)
    return {"Predicted Salary": prediction[0]}

# Directory structure should include a templates folder for HTML and static for CSS.
