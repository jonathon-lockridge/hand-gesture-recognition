from collections import deque
import torch


class GestureSmoother:
    """Averages the last `window` probability vectors so one noisy frame
    can't flip the label. Returns (class_index, confidence) or (None, conf)
    when the smoothed confidence is below `threshold`."""

    def __init__(self, window=8, threshold=0.7):
        self.threshold = threshold
        self.history = deque(maxlen=window)

    def update(self, probs):
        self.history.append(probs)
        smoothed = torch.stack(list(self.history)).mean(0)
        conf, idx = smoothed.max(0)
        if conf < self.threshold:
            return None, conf.item()
        return idx.item(), conf.item()

    def reset(self):
        """Call when the hand leaves the frame so stale frames don't linger."""
        self.history.clear()
