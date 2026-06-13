import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("monkey_model.keras")

classes = ['monkey', 'non-monkey']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Resize for prediction
    img = cv2.resize(frame, (128, 128))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model(img, training=False).numpy()

    predicted_class = classes[np.argmax(prediction)]
    confidence = float(np.max(prediction) * 100)

    # Attractive labels
    if confidence >= 50 and predicted_class == "monkey":
        label = "MONKEY DETECTED!"
        color = (0, 0, 255)   # Red

    elif confidence >= 50:
        label = "NON-MONKEY"
        color = (0, 255, 0)   # Green

    else:
        label = "ANALYZING..."
        color = (0, 255, 255) # Yellow

    # Header Box
    cv2.rectangle(frame, (0, 0), (700, 100), (0, 0, 0), -1)

    cv2.putText(
        frame,
        "AI MONKEY DETECTION SYSTEM",
        (20, 30),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        label,
        (20, 65),
        cv2.FONT_HERSHEY_DUPLEX,
        1.2,
        color,
        3
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.1f}%",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Monkey Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()