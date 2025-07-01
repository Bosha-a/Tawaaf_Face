import face_recognition
import numpy as np
import pickle
import faiss 
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


with open('names.pkl', 'rb') as file:
    known_names = pickle.load(file)

known_encodings = np.load('encodings.npy')

# إنشاء FAISS index
index = faiss.IndexFlatL2(128)
index.add(np.array(known_encodings, dtype='float32'))


@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    image = face_recognition.load_image_file(file.file)
    face_locations = face_recognition.face_locations(image)

    if len(face_locations) == 0:
        return JSONResponse(content={
            "found_faces": 0,
            "message": "No faces Matched"
        }, status_code=404)

    # extracting encodings 
    encoding = face_recognition.face_encodings(image, face_locations)[0]

    # Searching in database using Faiss
    D, I = index.search(np.array([encoding], dtype='float32'), 1)
    closest_match_index = I[0][0]
    distance = D[0][0]

    if distance < 0.35:
        matched_name = known_names[closest_match_index]
        return JSONResponse(content={
        "found_faces": 1,
        "matched_name": matched_name,
        "distance": float(distance)
    })

    else:
        return JSONResponse(content={
        "found_faces": 0,
        "matched_name": "No Face Matched",
    })

    