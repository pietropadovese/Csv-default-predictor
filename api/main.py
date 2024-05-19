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
import seaborn as sns
from pydantic import BaseModel



app = FastAPI(
    title="Loan default predictor",
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
        <title>Loan default predictor</title>
    </head>
    <body>
        <h1>Upload CSV file for prediction</h1>
        <form action="/predict_csv/" method="post" enctype="multipart/form-data">
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
    
    
@app.post("/predict_csv/")
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
    
    
class Company(BaseModel):
    gross_margin_ratio: float
    core_income_ratio: float
    cash_asset_ratio: float
    consolidated_liabilities_ratio: float
    tangible_assets_ratio: float
    revenues: float
    
    

@app.post("/predict_json/")
def predict(companies: List[Company]) -> List[str]:
    try:
        X = pd.DataFrame([dict(company) for company in companies])
        y_pred = model.predict(X)
        return list(y_pred)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV file: {e}")
        



    
@app.post("/visualize/", response_class=HTMLResponse)
async def visualize(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV file: {e}")
    
    try:    
        plot_files = []
        for col in df.columns:
            fig, ax = plt.subplots(ncols = 2, figsize = (10,5))
            sns.kdeplot(data = df, x = col, ax = ax[0])
            sns.boxplot(data = df, x = col, ax = ax[1], showfliers = False)
            fig.suptitle(col)
            ax[0].set_title('Kernel density')
            ax[1].set_title('Boxplot')
            plot_filename = f"static/{col}.png"
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