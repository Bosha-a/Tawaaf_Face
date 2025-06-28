import face_recognition
import numpy as np
import pickle
import faiss 
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.responses import JSONResponse

app = FastAPI()

with open('names1.pkl', 'rb') as file:
    known_names = pickle.load(file)

known_encodings = np.load('encodings1.npy')

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
            "message": "No faces detected"
        }, status_code=404)

    # extracting encodings 
    encoding = face_recognition.face_encodings(image, face_locations)[0]

    # Searching in database using Faiss
    D, I = index.search(np.array([encoding], dtype='float32'), 1)
    closest_match_index = I[0][0]
    distance = D[0][0]

    if distance < 0.3:
        matched_name = known_names[closest_match_index]
    else:
        matched_name = None

    return JSONResponse(content={
        "found_faces": 1,
        # "encoding": encoding.tolist(),
        "matched_name": matched_name,
        "distance": float(distance)
    })























#     # البحث عن أقرب وجه
#     D, I = index.search(np.array([encoding], dtype='float32'), 1)
#     closest_match_index = I[0][0]
#     distance = D[0][0]
#     print(f"Distance: {distance}")

#     if distance < req.threshold:
#         return {"matched_name": known_names[closest_match_index], "distance": distance}
#     else:
#         return {"matched_name": None, "distance": distance}

# # @app.post("/update_encodings")
# def update_encodings():
#     updated_names = []
#     skipped_names = []
#     failed_names = []


#     for person in collection.find({"encoding2": {"$exists": False}}):
#         name = person.get("name")
#         photo_url = person.get("photo")

#         if not name or not photo_url:
#             skipped_names.append(name or "Unknown")
#             continue

#         try:
#             response = requests.get(photo_url)
#             response.raise_for_status()
#             image = Image.open(BytesIO(response.content)).convert("RGB")
#             image_np = np.array(image)

#             face_locations = face_recognition.face_locations(image_np)
#             encoding_list = face_recognition.face_encodings(image_np, face_locations)

#             if encoding_list:
#                 encoding = encoding_list[0].tolist()

#                 collection.update_one(
#                     {"_id": person["_id"]},
#                     {"$set": {"encoding2": encoding}}
#                 )

#                 known_encodings.append(encoding)
#                 known_names.append(name)
#                 updated_names.append(name)
#             else:
#                 failed_names.append(name)

#         except Exception as e:
#             failed_names.append(f"{name} (error: {e})")


#     if updated_names:
#         np.save("encodings.npy", np.array(known_encodings, dtype='float32'))
#         with open("names.pkl", "wb") as f:
#             pickle.dump(known_names, f)

#     return {
#         "✅ updated": updated_names,
#         "⏭️ skipped (missing data)": skipped_names,
#         "❌ failed (no face or error)": failed_names
#     }