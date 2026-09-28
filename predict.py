import cv2
import torch
from hand_utils import create_landmarker, detect, to_feature_vector, draw_hand
from model import GestureNet
from smoothing import GestureSmoother

MODEL_PATH = "models/gesture_model.pt"
WINDOW = 8            # frames to average over
THRESHOLD = 0.7       # min smoothed confidence to show a label

# ---- load the trained model ----
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
classes = checkpoint["classes"]
model = GestureNet(n_features=checkpoint["n_features"], n_classes=len(classes))
model.load_state_dict(checkpoint["state_dict"])
model.eval()

landmarker = create_landmarker()
cap = cv2.VideoCapture(0)
smoother = GestureSmoother(window=WINDOW, threshold=THRESHOLD)

def draw_label(frame, text, color):
    """Text with a dark outline so it's readable over any background."""
    k = frame.shape[0] / 480                      # scale everything with frame height
    pos = (int(16 * k), int(52 * k))
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, text, pos, font, 1.4 * k, (0, 0, 0), int(6 * k), cv2.LINE_AA)   # outline
    cv2.putText(frame, text, pos, font, 1.4 * k, color, int(2 * k), cv2.LINE_AA)       # fill

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
        idx, conf = smoother.update(probs)

        if idx is not None:
            text = f"{classes[idx]}  {conf * 100:.0f}%"
            color = (255, 160, 0)     # electric blue (BGR)
        else:
            text = "..."
            color = (0, 200, 255)
        draw_label(frame, text, color)
    else:
        smoother.reset()               # hand left the frame, forget old predictions

    cv2.imshow("gesture", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()