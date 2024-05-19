import pickle
import pandas as pd
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, StreamingResponse
import io
import matplotlib.pyplot as plt
from fastapi.staticfiles import StaticFiles
import os



app = FastAPI(
    title="Zoo Animal CLassification",
    version="0.1.0",
)

# Create a directory for storing uploaded files and static files
os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    raise

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
        <h1>Upload CSV file to visualize</h1>
        <form action="/visualize/" method="post" enctype="multipart/form-data">
            <input type="file" name="file"><br><br>
            <input type="submit" value="Visualize">
        </form>
    </body>
    </html>
    """, media_type="text/html")
    
    
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
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
    
    except Exception as e:
        return {"error": str(e)}

    
@app.post("/visualize/", response_class=HTMLResponse)
async def visualize(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        print(type(contents))
        df = pd.read_csv(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV file: {e}")
    
    try:    
        plot_files = []
        for column in df.columns:
            plt.figure()
            df[column].value_counts().plot(kind='bar')
            plot_filename = f"static/{column}.png"
            plt.savefig(plot_filename)
            plt.close()
            plot_files.append(plot_filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating plots: {e}")
    
    try:    
        plot_html = ""
        for plot_file in plot_files:
            plot_html += f'<img src="/{plot_file}" alt="{plot_file}"><br>'   
        return HTMLResponse(content=f"""
        <html>
        <head>
            <title>CSV Visualization</title>
        </head>
        <body>
            <h1>Barplots for CSV columns</h1>
            {plot_html}
        </body>
        </html>
        """, media_type="text/html")
    
    except Exception as e:
        raise HTTPException(status_code=600, detail=f"Error displaying plots: {e}")
        

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)