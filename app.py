from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import os
import uuid

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLOv8 model sekali saat awal
model = YOLO("best.pt")

@app.route('/predict', methods=['POST'])
def predict():
    print("📥 Menerima permintaan dari Flutter...")

    if 'image' not in request.files:
        print("❌ Tidak ada gambar ditemukan.")
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files['image']
    filename = f"{uuid.uuid4()}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    image.save(filepath)
    print(f"📂 Gambar disimpan: {filepath}")

    try:
        # Prediksi dengan model
        results = model.predict(source=filepath, conf=0.3)

        # Ambil hasil prediksi
        pred = results[0]
        boxes = pred.boxes
        names = model.names

        detected_objects = []
        for box in boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

            detected_objects.append({
                "class": names[cls_id],
                "confidence": round(confidence, 2),
                "bbox": [round(x, 2) for x in bbox]
            })

        print(f"✅ Deteksi selesai: {len(detected_objects)} objek ditemukan.")

        return jsonify({
            "message": "Prediction success",
            "detections": detected_objects
        })

    except Exception as e:
        print(f"❌ Error saat prediksi: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        # Hapus file setelah diproses untuk hemat storage
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5000)

