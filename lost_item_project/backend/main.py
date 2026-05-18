from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from database import engine
from models import Base  
from dotenv import load_dotenv
from google.cloud import vision
from routers import lost_items, found_items
import models
import os

load_dotenv()

app = FastAPI()

Base.metadata.create_all(bind=engine)

os.makedirs("static/images", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(lost_items.router)
app.include_router(found_items.router)

@app.get("/")
def read_root():
    return {"message": "Lost item API is running"}

@app.get("/test-vision")
def test_vision():
    client = vision.ImageAnnotatorClient()
    return {"message": "Google Vision API 연동 성공!"}