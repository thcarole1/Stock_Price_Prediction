
from Stock_Price_Prediction_package_folder.api_functions.predict import predict_stock_price_api
from fastapi import FastAPI
from fastapi.responses import FileResponse
from io import BytesIO
import pandas as pd
import zipfile

app = FastAPI()

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'New project': 'This is the first app of my new project !'}

@app.get("/ticker/")
async def read_item(query: str):
    image_dict, summary = predict_stock_price_api(query)

    # Create a ZIP archive in memory
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        # Add images
        for value in image_dict.values():
            image_name = value[0]
            image_data = value[1] # Replace with actual image data
            zip_file.writestr(image_name, image_data)

        # Add Pandas dataframe as CSV
        csv_buffer = BytesIO()
        summary.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        zip_file.writestr("summary.csv", csv_buffer.read())

    zip_buffer.seek(0)  # Reset buffer cursor for reading

    # Save the BytesIO to an actual file
    with open("zip_buffer_file", "wb") as f:
        f.write(zip_buffer.getvalue())

    return FileResponse("zip_buffer_file", media_type="application/zip", filename="data.zip")
