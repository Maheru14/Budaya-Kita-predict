# API DOCUMENTATION

### 1. **Image Prediction**

**URL:** `POST /predict-image`  
**Endpoint:** [https://prediksi-budaya-592544960467.asia-southeast2.run.app/predict-image](https://prediksi-budaya-592544960467.asia-southeast2.run.app/predict-image)  
**Request:**

```json
{
  "file": file,
  "user_id": string
}
```

**Response:**

```json
{
  "confidence": float,
    "created_at": datetime,
    "deskripsi": string,
    "file_url": string,
    "prediction": string,
    "user_id": string
}
```

---

### 2. **Prediction History**

**URL:** `GET /prediction-history?user_id=[user_id]`  
**Endpoint:** [https://prediksi-budaya-592544960467.asia-southeast2.run.app/prediction-history?user_id=[user_id]](https://prediksi-budaya-592544960467.asia-southeast2.run.app/prediction-history?user_id=0)  
**Request:**

```json
{
  "user_id": string,
}
```

**Response:**

```json
{
    "history": [
        {
            "created_at": datetime,
            "deskripsi": string,
            "file_url": string,
            "filename": string,
            "label_name": string,
            "user_id": string
        },
    ]
}
```

---
