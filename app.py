from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from pymongo import MongoClient
from bson import ObjectId
import requests
from pathlib import Path
import os
import uuid
from zoneinfo import ZoneInfo 
from datetime import datetime, timedelta
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity, create_access_token

app = Flask(__name__)
CORS(app)

app.config['JWT_SECRET_KEY'] = 'fridgeeye'
app.config['JWT_LEEWAY'] = timedelta(seconds=120)
jwt = JWTManager(app)

SHELF_LIFE = {
    'apel': 6,
    'wortel': 7,
    'tomat': 5,
    'pisang': 4,
    'semangka': 1
}

MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)
db = client["predict"]
fruits_collection = db["fruits"]

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_URL = "https://drive.google.com/uc?export=download&id=1fABCdefGhIjKlmn"
model_path = Path("best.pt")

if not model_path.exists():
    print("⬇️ Mengunduh model YOLO...")
    r = requests.get(MODEL_URL, stream=True)
    with open(model_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("✅ Unduhan selesai")

# Load model
from ultralytics import YOLO
model = YOLO(str(model_path))

@app.route('/predict', methods=['POST'])
def predict():
    print("Menerima permintaan dari Flutter...")

    if 'image' not in request.files:
        print("❌ Tidak ada gambar ditemukan.")
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files['image']
    filename = f"{uuid.uuid4()}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    image.save(filepath)
    print(f"📂 Gambar disimpan: {filepath}")

    try:
        results = model.predict(source=filepath, conf=0.3)

        pred = results[0]
        boxes = pred.boxes
        names = model.names

        detected_objects = []
        for box in boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].tolist()  

            detected_objects.append({
                "class": names[cls_id],
                "confidence": round(confidence, 2),
                "bbox": [round(x, 2) for x in bbox]
            })

        print(f"Deteksi selesai: {len(detected_objects)} objek ditemukan.")

        return jsonify({
            "message": "Prediction success",
            "detections": detected_objects
        })

    except Exception as e:
        print(f"Error saat prediksi: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/fruits', methods=['POST'])
@jwt_required()
def add_fruit():
    current_user = get_jwt_identity()
    data = request.get_json()

    name = data.get('name', '').lower()
    image = data.get('image')
    purchase_date_str = data.get('purchaseDate')

    if not name or not purchase_date_str:
        return jsonify({'error': 'Nama dan tanggal pembelian wajib diisi'}), 400

    try:
        purchase_date = datetime.fromisoformat(purchase_date_str).replace(tzinfo=ZoneInfo("Asia/Jakarta"))  # ✅ WIB
    except Exception:
        return jsonify({'error': 'Format tanggal tidak valid'}), 400

    life = SHELF_LIFE.get(name, 5)
    expiry_date = purchase_date + timedelta(days=life)

    fruit = {
        'user_email': current_user,
        'name': name,
        'image': image,
        'purchaseDate': purchase_date.isoformat(),
        'expiryDate': expiry_date.isoformat()
    }

    fruits_collection.insert_one(fruit)
    return jsonify({'message': 'Buah berhasil ditambahkan'}), 201

@app.route('/fruits', methods=['GET'])
@jwt_required()
def get_fruits():
    current_user = get_jwt_identity()

    fruits = fruits_collection.find({'user_email': current_user})
    all_fruits = []

    for buah in fruits:
        buah['_id'] = str(buah['_id'])
        buah['purchaseDate'] = str(buah['purchaseDate'])
        buah['expiryDate'] = str(buah['expiryDate'])
        all_fruits.append(buah)

    return jsonify(all_fruits), 200

@app.route('/notifications', methods=['GET'])
@jwt_required()
def get_notifications():
    current_user = get_jwt_identity()
    now = datetime.now(ZoneInfo("Asia/Jakarta")) 

    busuk = fruits_collection.find({
        'user_email': current_user,
        'expiryDate': {'$lt': now.isoformat()}
    })

    hampir_busuk = fruits_collection.find({
        'user_email': current_user,
        'expiryDate': {
            '$gte': now.isoformat(),
            '$lte': (now + timedelta(days=2)).isoformat()
        }
    })

    notif = {
        'sudah_busuk': [fruit['name'] for fruit in busuk],
        'hampir_busuk': [fruit['name'] for fruit in hampir_busuk]
    }

    return jsonify(notif), 200

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5000)

