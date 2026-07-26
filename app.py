from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import shutil
import os
import uuid

app = FastAPI()

model = YOLO("best.pt")

label_map = {
    "sinhala_ak": "අ",
    "sinhala_ba": "බ",
    "sinhala_dha": "ද",
    "sinhala_ga": "ග",
    "sinhala_ha": "හ",
    "sinhala_i": "ඉ",
    "sinhala_ka": "ක",
    "sinhala_la": "ල",
    "sinhala_ma": "ම",
    "sinhala_na": "න",
    "sinhala_pa": "ප",
    "sinhala_ra": "ර",
    "sinhala_sa": "ස",
    "sinhala_ta": "ට",
    "sinhala_tha": "ත",
    "sinhala_u": "උ",
    "sinhala_ya": "ය",
}

UPLOAD_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    file_path = os.path.join(
        UPLOAD_FOLDER,
        f"{uuid.uuid4()}.jpg"
    )

    # Save  image
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # YOLO prediction
    results = model(file_path)

    output = []

    for r in results:

        if r.boxes is None or len(r.boxes) == 0:
            continue

        for box in r.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            width = x2 - x1
            height = y2 - y1

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Ignore low confidence
            if confidence < 0.3:
                continue

            # Use YOLO trained class names
            class_name = model.names[class_id]

            # Convert Sinhala label
            sinhala_letter = label_map.get(
                class_name,
                class_name
            )

            print(
                "CLASS ID:",
                class_id,
                "YOLO NAME:",
                class_name,
                "SINHALA:",
                sinhala_letter,
                "CONF:",
                confidence
            )

            output.append({
                "label": class_name,
                "sinhala": sinhala_letter,
                "confidence": confidence,

                "x": center_x,
                "y": center_y,

                "width": width,
                "height": height
            })


    # Remove temporary image
    # if os.path.exists(file_path):
    #     os.remove(file_path)

    if len(output) == 0:
        return {
            "success": False,
            "message": "No handwriting detected. Try drawing clearer.",
            "predictions": []
        }

    # Highest confidence first
    output.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    print("WORD PREDICTIONS:", output)

    return {
        "success": True,
        "predictions": output
    }