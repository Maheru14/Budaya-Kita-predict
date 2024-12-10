from datetime import datetime
from flask import Flask, request, jsonify
from google.cloud import storage, firestore
import pytz
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import os

app = Flask(__name__)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/dianmaheru/Documents/Prediksi_budaya/budayakita-690e5dcb36ab.json"
BUCKET_NAME = "bucket-budayakita"
db = firestore.Client(database="budayakitadb")

interpreter = tf.lite.Interpreter(model_path="/Users/dianmaheru/Documents/Prediksi_budaya/model_quant_another (1).tflite")
interpreter.allocate_tensors()

labels = {
    0: "Batik Celup",
    1: "Batik Cendrawasih",
    2: "Batik Dayak",
    3: "Batik Geblek Renteng",
    4: "Batik Insang",
    5: "Batik Kawung",
    6: "Batik Lasem",
    7: "Batik Megamendung",
    8: "Batik Parang",
    9: "Batik Poleng",
    10: "Batik Pring",
    11: "Batik Sekar",
    12: "Batik Sidoluhur",
    13: "Batik Tambal",
    14: "Batik Truntum"
}
deskripsiList = {
    0: "Batik Celup adalah jenis batik yang dihasilkan melalui teknik pencelupan kain ke dalam larutan warna tertentu. Teknik ini sering dikaitkan dengan motif abstrak atau corak sederhana. Batik ini populer di berbagai daerah, terutama di Jawa.",
    1: "Batik Cendrawasih berasal dari Papua, menggambarkan keindahan burung Cendrawasih yang menjadi simbol kebanggaan wilayah tersebut. Motif ini sering memadukan warna cerah dan elemen alam Papua.",
    2: "Batik Dayak berasal dari Kalimantan dengan motif yang terinspirasi dari seni ukir suku Dayak. Motifnya sering berupa pola geometris atau gambar flora dan fauna khas Kalimantan.",
    3: "Batik Geblek Renteng berasal dari Kulon Progo, Yogyakarta. Motifnya terinspirasi dari makanan khas setempat bernama geblek dan memiliki pola rantai (renteng) yang melambangkan persatuan dan kerja sama.",
    4: "Batik Insang berasal dari Kalimantan Timur. Motifnya terinspirasi dari insang ikan, yang menggambarkan kedekatan masyarakat setempat dengan budaya maritim.",
    5: "Batik Kawung adalah salah satu motif batik tertua di Indonesia, berasal dari Jawa. Motifnya menyerupai biji buah kawung (aren), melambangkan kesucian dan kebijaksanaan.",
    6: "Batik Lasem berasal dari daerah Lasem, Jawa Tengah. Motifnya dipengaruhi oleh budaya Tionghoa dengan warna-warna cerah seperti merah dan biru, serta pola seperti bunga dan naga.",
    7: "Batik Megamendung berasal dari Cirebon. Motifnya menyerupai awan mendung dengan gradasi warna. Motif ini melambangkan ketenangan dan kesejukan.",
    8: "Batik Parang adalah motif khas Jawa yang melambangkan perjuangan dan semangat pantang menyerah. Motifnya menyerupai garis-garis diagonal seperti ombak laut.",
    9: "Batik Poleng berasal dari Bali, dengan motif kotak-kotak hitam dan putih. Motif ini melambangkan keseimbangan antara dua kekuatan, baik dan buruk.",
    10: "Batik Pring berasal dari Magetan, Jawa Timur. Motifnya menggambarkan bambu (pring) yang melambangkan kehidupan sederhana dan kekuatan.",
    11: "Batik Sekar adalah motif batik dengan pola bunga (sekar) yang melambangkan keindahan dan kesuburan. Motif ini ditemukan di berbagai daerah di Jawa.",
    12: "Batik Sidoluhur berasal dari Solo, Jawa Tengah. Motif ini melambangkan harapan untuk mencapai kehidupan yang luhur dan penuh kebajikan.",
    13: "Batik Tambal adalah motif batik yang terdiri dari berbagai pola kecil yang disusun menjadi satu. Motif ini melambangkan usaha untuk memperbaiki diri dan kehidupan.",
    14: "Batik Truntum adalah motif batik dari Solo yang diciptakan oleh permaisuri Raja Paku Buwono III. Motif ini melambangkan cinta yang tumbuh kembali dan hubungan yang harmonis."
}

def format_label_folder(label):
    return label.lower().replace(" ", "_")

def preprocess_image(image_file):
    try:
        img = Image.open(image_file).convert('RGB') 
        img = img.resize((224, 224)) 
        img_array = np.array(img) / 255.0 
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32) 
        return img_array
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

@app.route('/predict-image', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not request.form.get("user_id"):
        return jsonify({"error": "No user id"}), 400

    try:
        file.seek(0) 

        input_data = preprocess_image(file)

        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_index)

        predicted_label = np.argmax(predictions)
        confidence = predictions[0][predicted_label]
        label_name = labels[predicted_label]
        deskripsi = deskripsiList[predicted_label]

        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"history/{format_label_folder(label_name)}/{file.filename}")
        file.seek(0) 
        blob.upload_from_file(file)
        
        jakarta_timezone = pytz.timezone("Asia/Jakarta")
        created_at = datetime.now(jakarta_timezone).isoformat()

        file_url = blob.public_url

        doc_ref = db.collection('predictions').document()
        doc_ref.set({
            "label_name": label_name,
            "deskripsi": deskripsi,
            "file_url": file_url,
            "filename": file.filename,
            "user_id": request.form.get("user_id"),
            "created_at": created_at
        })

        return jsonify({
            "prediction": label_name,
            "deskripsi": deskripsi,
            "confidence": float(confidence),
            "file_url": file_url,
            "user_id": request.form.get("user_id"),
            "created_at": created_at
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/prediction-history', methods=['GET'])
def prediction_history():
    try:
        user_id = request.args.get('user_id')

        if not user_id:
            return jsonify({"error": "Parameter 'user_id' is required"}), 400

        predictions_ref = db.collection('predictions')
        docs = predictions_ref.stream()

        history = []
        for doc in docs:
            data = doc.to_dict()
            if data.get("user_id") == user_id:
                history.append({
                    "user_id": data.get("user_id"),
                    "filename": data.get("filename"),
                    "file_url": data.get("file_url"),
                    "label_name": data.get("label_name"),
                    "created_at": data.get("created_at")
                })

        history.sort(key=lambda x: x.get("created_at"))

        return jsonify({"history": history}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
