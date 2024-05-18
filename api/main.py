import pickle
import pandas as pd
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, StreamingResponse
import io

app = FastAPI(
    title="Zoo Animal CLassification",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

model = pickle.load(
    open('model.pkl', 'rb')
)

@app.get("/", response_class = HTMLResponse)
def home():
    return HTMLResponse(content="""
     <html>
    <head>
        <title>Animal Predictor</title>
    </head>
    <body>
        <h1>Upload CSV file for prediction</h1>
        <form action="/predict/" method="post" enctype="multipart/form-data">
            <input type="file" name="file"><br><br>
            <input type="submit" value="Upload">
        </form>
    </body>
    </html>
    """, media_type="text/html")
    
    
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded CSV file
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

    # Assume the DataFrame has the same structure as the training data
    predictions = model.predict(df)

    # Add predictions to the DataFrame
    df['predictions'] = predictions

    # Convert DataFrame to CSV
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(output, media_type="text/csv", headers={"Content-Disposition": "attachment;filename=predictions.csv"})