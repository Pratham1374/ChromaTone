import numpy as np
from deepface import DeepFace

def analyze_gender_deepface(pil_image):
    """
    Uses the DeepFace RetinaFace logic to detect visibility and gender.
    Replaces the old gender_rf model.
    """
    try:
        # Convert PIL Image to BGR (OpenCV format) as required by DeepFace
        img_array = np.array(pil_image)
        frame = img_array[:, :, ::-1].copy() 

        # --- LOGIC FROM GENDER_DETECTOR.PY ---
        # 'retinaface' is used for high accuracy. 
        # 'enforce_detection=True' handles obscured faces by raising a ValueError.
        results = DeepFace.analyze(
            img_path=frame, 
            actions=['gender'], 
            detector_backend='retinaface', 
            enforce_detection=True,
            silent=True
        )

        # Process results
        res = results[0] if isinstance(results, list) else results
        gender_label = res.get('dominant_gender', "Unknown")
        
        # Normalize gender label for your existing frontend mapping
        gender_label = "Man" if gender_label.lower() == "man" else "Woman"
        
        conf_score = res.get('gender', {}).get(gender_label, 0)

        # Apply your 60% confidence logic
        if conf_score < 60.0:
            return f"Uncertain ({gender_label}?)", {"confidence": conf_score}
            
        return gender_label, {"confidence": conf_score}

    except ValueError:
        # This catches hands covering face or non-face images
        # Returns "Face Obscured" to trigger your backend's Guard Clause
        return "Face Obscured", {}
    except Exception as e:
        print(f"❌ Gender Analysis Error: {str(e)}")
        return "Unknown", {}