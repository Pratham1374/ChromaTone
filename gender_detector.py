import cv2
from deepface import DeepFace

def get_gender_prediction(frame):
    """
    Takes a single frame (BGR image) and returns gender and confidence.
    """
    try:
        # We use 'retinaface' as in your provided code for accuracy.
        # 'enforce_detection=True' handles your requirement for obscured faces.
        results = DeepFace.analyze(
            img_path=frame, 
            actions=['gender'], 
            detector_backend='retinaface', 
            enforce_detection=True,
            silent=True
        )

        res = results[0] if isinstance(results, list) else results
        gender_label = res.get('dominant_gender', "Unknown")
        conf_score = res.get('gender', {}).get(gender_label, 0)

        # Apply your 60% confidence logic
        if conf_score < 60.0:
            return f"Uncertain ({gender_label}?)", conf_score
        return gender_label, conf_score

    except ValueError:
        # Triggers for hands covering face or non-face images
        return "No face detected", 0
    except Exception as e:
        return f"Error: {str(e)}", 0