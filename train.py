import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from model import GestureNet

DATA_PATH = "data/gestures.csv"
MODEL_PATH = "models/gesture_model.pt"
EPOCHS = 60
BATCH_SIZE = 64

# ---- load data ----
raw = np.genfromtxt(DATA_PATH, delimiter=",", skip_header=1, dtype=str)
labels_str = raw[:, 0]
X = raw[:, 1:].astype(np.float32)

classes = sorted(str(c) for c in set(labels_str))                 # ['fist', 'palm', ...]
y = np.array([classes.index(l) for l in labels_str])
print(f"{len(X)} samples, {len(classes)} classes: {classes}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- move to device ----
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"training on {device}")
X_train_t = torch.tensor(X_train).to(device)
y_train_t = torch.tensor(y_train).to(device)
X_test_t = torch.tensor(X_test).to(device)
y_test_t = torch.tensor(y_test).to(device)

# ---- train ----
model = GestureNet(n_features=X.shape[1], n_classes=len(classes)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(X_train_t))
    for i in range(0, len(perm), BATCH_SIZE):
        idx = perm[i:i + BATCH_SIZE]
        logits = model(X_train_t[idx])
        loss = loss_fn(logits, y_train_t[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            acc = (model(X_test_t).argmax(1) == y_test_t).float().mean().item()
        print(f"epoch {epoch + 1:3d}   loss {loss.item():.4f}   test acc {acc:.3f}")

# ---- evaluate ----
model.eval()
with torch.no_grad():
    preds = model(X_test_t).argmax(1).cpu().numpy()
print()
print(classification_report(y_test, preds, target_names=classes))

# ---- save ----
torch.save(
    {"state_dict": model.state_dict(), "classes": classes, "n_features": X.shape[1]},
    MODEL_PATH,
)
print(f"saved model to {MODEL_PATH}")