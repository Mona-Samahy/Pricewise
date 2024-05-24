from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import pickle
import uvicorn
from pydantic import BaseModel, Field
from typing import List, Annotated
from contextlib import asynccontextmanager
import base64

async def decode_image(image_data: str) -> Image.Image:
    try:
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading image: {str(e)}")

model, dataset_embeddings, dataset_image_paths = None, None, None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, dataset_embeddings, dataset_image_paths
    model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    with open("embeddings.pkl", "rb") as f:
        dataset_embeddings = pickle.load(f)

    with open("filenames.pkl", "rb") as f:
        dataset_image_paths = pickle.load(f)

    yield

app = FastAPI(lifespan=lifespan)

class UserRequest(BaseModel):
    image_data: str = Field(...)
    top_n: int = Field(default=5)

class PredictionResult(BaseModel):
    result: List [str]

@app.post("/predict")
async def predict(req_data: Annotated[UserRequest, Body(embed=False)]):
    try:
        top_n = req_data.top_n
        print ("###############################") 
        print (top_n)
        img = await decode_image(req_data.image_data)
    
        img = img.resize((256, 256))
        img_array = np.array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        query_embedding = model.predict(img_array)

        similarities = cosine_similarity(query_embedding, dataset_embeddings)

        top_n_indices = np.argsort(similarities[0])[::-1][:top_n]

        similar_image_ids = [dataset_image_paths[i] for i in top_n_indices]

        similar_image_ids = [
            os.path.splitext(os.path.basename(path))[0] for path in similar_image_ids
        ]

        # lastResult = [] 

        # for img    in similar_image_ids  : 
        #     img_id = img 
        #     img_path =  os.path.join(  'images' , img  + '.jpg'  )
        #     # print(img_path) 

        #     if not os.path.exists( img_path ) : 
        #         raise HTTPException(status_code=400, detail={"error":  f"cannot find this file path {img_path}"})
            
        #     with open(img_path ,  'rb') as f  :
        #         encoded_img   = base64.b64encode(  f.read( )  )  
        #         lastResult.append (  {img_id : encoded_img  } )
                
        
        # print(lastResult)

        return PredictionResult(result=similar_image_ids)
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app )
