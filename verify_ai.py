import cv2
from gender_detector import get_gender_prediction
# import your_skintone_module (Commented out for now)

def process_frame(frame_path):
    img = cv2.imread(frame_path)
    if img is None:
        print(f"❌ Error: Could not find image at {frame_path}")
        return

    # 1. Run the new Gender/Visibility Detection logic
    gender, confidence = get_gender_prediction(img)

    # 2. Results
    print("-" * 30)
    print(f"FILE: {frame_path}")
    print(f"RESULT: {gender}")
    print(f"CONFIDENCE: {confidence:.1f}%")
    print("-" * 30)

# TEST 1: The Obscured Face (The one giving you "Light Beige")
# Make sure you have an image with this name in the backend folder
process_frame("hands_covered.jpg") 

# TEST 2: A Clear Face
# process_frame("clear_face.jpg")