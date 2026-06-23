import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
import time
import pyttsx3

# ==========================
# TEXT TO SPEECH
# ==========================
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# ==========================
# LOAD MODEL
# ==========================
model = tf.keras.models.load_model("monkey_model.keras")

classes = ['monkey', 'non-monkey']

# ==========================
# CAMERA
# ==========================
cap = cv2.VideoCapture(0)

# ==========================
# SCAN SETTINGS
# ==========================
scan_duration = 3
scan_start = time.time()

predictions = []
confidences = []

final_result = None
final_confidence = 0

alert_played = False

# ==========================
# MAIN LOOP
# ==========================
while True:

    ret, frame = cap.read()

    if not ret:
        break

    h, w, _ = frame.shape

    # ==========================
    # PREPROCESS IMAGE
    # ==========================
    img = cv2.resize(frame, (128, 128))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    predicted_class = classes[np.argmax(prediction)]
    confidence = float(np.max(prediction) * 100)

    # ==========================
    # HEADER PANEL
    # ==========================
    cv2.rectangle(
        frame,
        (0, 0),
        (w, 120),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        frame,
        "AI MONKEY DETECTION SYSTEM",
        (20, 40),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Machine Learning University Project",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (180, 180, 180),
        2
    )

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cv2.putText(
        frame,
        current_time,
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    # ==========================
    # SCANNING MODE
    # ==========================
    elapsed = time.time() - scan_start

    if elapsed < scan_duration:

        predictions.append(predicted_class)
        confidences.append(confidence)

        progress = elapsed / scan_duration

        # Yellow Border
        cv2.rectangle(
            frame,
            (0, 0),
            (w - 1, h - 1),
            (0, 255, 255),
            8
        )

        cv2.putText(
            frame,
            "SCANNING...",
            (120, 250),
            cv2.FONT_HERSHEY_DUPLEX,
            2,
            (0, 255, 255),
            4
        )

        # Progress Bar Outline
        cv2.rectangle(
            frame,
            (100, 320),
            (550, 360),
            (255, 255, 255),
            2
        )

        # Progress Fill
        cv2.rectangle(
            frame,
            (100, 320),
            (100 + int(450 * progress), 360),
            (0, 255, 255),
            -1
        )

        percent = int(progress * 100)

        cv2.putText(
            frame,
            f"{percent}%",
            (280, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

    else:

        # ==========================
        # FINAL DECISION
        # ==========================
        if final_result is None:

            monkey_count = predictions.count("monkey")
            non_monkey_count = predictions.count("non-monkey")

            final_confidence = np.mean(confidences)

            if monkey_count > non_monkey_count:

                final_result = "MONKEY DETECTED"
                border_color = (0, 0, 255)
                text_color = (0, 0, 255)

                if not alert_played:
                    engine.say("Warning. Monkey detected.")
                    engine.runAndWait()
                    alert_played = True

            else:

                final_result = "NON MONKEY"
                border_color = (0, 255, 0)
                text_color = (0, 255, 0)

                if not alert_played:
                    engine.say("No monkey detected.")
                    engine.runAndWait()
                    alert_played = True

        # Keep colors after detection
        if final_result == "MONKEY DETECTED":
            border_color = (0, 0, 255)
            text_color = (0, 0, 255)
        else:
            border_color = (0, 255, 0)
            text_color = (0, 255, 0)

        # Border
        cv2.rectangle(
            frame,
            (0, 0),
            (w - 1, h - 1),
            border_color,
            8
        )

        # Result Box
        cv2.rectangle(
            frame,
            (50, 180),
            (620, 400),
            (30, 30, 30),
            -1
        )

        cv2.putText(
            frame,
            "FINAL RESULT",
            (190, 230),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            final_result,
            (90, 300),
            cv2.FONT_HERSHEY_DUPLEX,
            1.5,
            text_color,
            4
        )

        cv2.putText(
            frame,
            f"Confidence : {final_confidence:.1f}%",
            (140, 360),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Press R To Rescan",
            (170, 390),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

    # ==========================
    # FOOTER
    # ==========================
    cv2.putText(
        frame,
        "ESC = Exit",
        (w - 150, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # ==========================
    # SHOW WINDOW
    # ==========================
    cv2.imshow(
        "AI Monkey Detection Dashboard",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # ESC EXIT
    if key == 27:
        break

    # RESCAN
    if key == ord('r'):
        scan_start = time.time()
        predictions = []
        confidences = []
        final_result = None
        final_confidence = 0
        alert_played = False

# ==========================
# CLEANUP
# ==========================
cap.release()
cv2.destroyAllWindows()