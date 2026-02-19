from flask import Flask, request, jsonify
from flask_cors import CORS
import io
from PIL import Image
from deepface_wrapper import analyze_gender_deepface

app = Flask(__name__)
CORS(app)

@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        # Import models inside to keep startup fast
        from models.deep_skintone_wrapper import predict_skin_tone_deep
        # ✅ NEW: Import the helper that matches accessories/makeup to the skin tone
        from accessory_data import get_recommendations 
        
        pil = Image.open(io.BytesIO(file.read())).convert("RGB")

        # 1. AI GATEKEEPER (RetinaFace)
        gender_label, _ = analyze_gender_deepface(pil)
        
        # If RetinaFace fails to find a face (hands in the way)
        if gender_label == "Face Obscured" or gender_label == "No face detected":
            return jsonify({
                "status": "error",
                "message": "Please use an image where the face is clearly visible.",
                "tone": None,
                "gender": "Unknown"
            })

        # 2. SKIN TONE ANALYSIS
        tone_label, probs = predict_skin_tone_deep(pil)

        # 3. ✅ CONNECTION FIX: Fetch specific accessories and makeup for this tone/gender
        # This ensures the 'accessories' and 'makeup' tabs are no longer empty
        recommendations = get_recommendations(tone_label, gender_label)

        return jsonify({
            "status": "success",
            "tone": str(tone_label),
            "gender": str(gender_label),
            "probs": probs,
            "accessories": recommendations # ✅ Sending Watches, Jewelry, and Makeup data
        })

    except Exception as e:
        print(f"Server Error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    print("✅ Hue AI Server listening on http://127.0.0.1:8000")
    app.run(host="127.0.0.1", port=8000, debug=True)