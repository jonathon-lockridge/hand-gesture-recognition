import cv2
import torch
from hand_utils import create_landmarker, detect, to_feature_vector, draw_hand
from model import GestureNet

MODEL_PATH = "models/gesture_model.pt"

# ---- load the trained model ----
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
classes = checkpoint["classes"]
model = GestureNet(n_features=checkpoint["n_features"], n_classes=len(classes))
model.load_state_dict(checkpoint["state_dict"])
model.eval()

landmarker = create_landmarker()
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    landmarks = detect(landmarker, frame)

    if landmarks is not None:
        draw_hand(frame, landmarks)
        features = torch.tensor(to_feature_vector(landmarks)).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(features), dim=1)[0]
        conf, idx = probs.max(0)
        text = f"{classes[idx]}  {conf.item() * 100:.0f}%"
        cv2.putText(frame, text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("gesture", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()