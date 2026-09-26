import csv
import os
import sys
import cv2
from hand_utils import create_landmarker, detect, to_feature_vector, draw_hand

if len(sys.argv) != 2:
    print("usage: python collect_data.py <gesture_name>")
    sys.exit(1)

label = sys.argv[1]
out_path = "data/gestures.csv"
write_header = not os.path.exists(out_path)

landmarker = create_landmarker()
cap = cv2.VideoCapture(0)
recording = False
count = 0

with open(out_path, "a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(["label"] + [f"f{i}" for i in range(63)])

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)                 # mirror so it feels natural
        landmarks = detect(landmarker, frame)

        if landmarks is not None:
            draw_hand(frame, landmarks)
            if recording:
                writer.writerow([label] + to_feature_vector(landmarks).tolist())
                count += 1

        status = "RECORDING" if recording else "paused  [SPACE record / Q quit]"
        cv2.putText(frame, f"{label}: {count}  {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("collect", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            recording = not recording
        elif key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
print(f"saved {count} samples for '{label}' to {out_path}")