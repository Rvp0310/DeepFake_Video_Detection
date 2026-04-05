import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import cv2
import numpy as np
from pathlib import Path

# dataset class
def add_noise(x):
    if torch.rand(1) < 0.3:
        noise = torch.randn_like(x) * 0.05
        x = torch.clamp(x + noise, 0, 1)
    return x
    
class DeepfakeDataset(Dataset):
    
    def __init__(self, folder, mode = "train"):
        self.files = sorted(Path(folder).glob("*.pt"))
        self.mode = mode
        self.transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 0.2)),
            T.ColorJitter(brightness=0.2, contrast=0.2)
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])

        spatial = data["spatial"].float()
        frequency = data["frequency"].float()
        label = data["label"].float()

        if self.mode == "train":
            spatial = torch.stack([add_noise(self.transform(f)) for f in spatial])

        return {
            "spatial": spatial,
            "frequency": frequency,
            "label": label
        }


# frame extraction function
def extract_frames(vid_path, num_frames = 16, resize=None):
    cap = cv2.VideoCapture(str(vid_path))

    if not cap.isOpened():
        print(f"Error opening video: {vid_path}")
        return

    frames = []
    prev_gray = None
    diffs = []
    all_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if resize is not None:
            frame = cv2.resize(frame, resize)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        all_frames.append(frame)

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            score = np.sum(diff)
            diffs.append(score)
        else:
            diffs.append(0)

        prev_gray = gray

    cap.release()

    idxs = np.argsort(diffs)[-num_frames:]
    idxs = sorted(idxs)

    selected = [all_frames[i] for i in idxs]
    return selected


# face detection function
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def face_detect_trace(vid_path):
    faces = []

    frames = extract_frames(vid_path)
    
    best_box = None
    best_frame_idx = 0
    max_area = 0
    
    for i, frame in enumerate(frames[:5]):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = face_detector.detectMultiScale(gray, 1.3, 5)
    
        for (x,y,w,h) in detected:
            area = w * h
            if area > max_area:
                max_area = area
                best_box = (x,y,w,h)
                best_frame_idx = i
    
    if best_box is None:
        return [cv2.resize(f, (224,224)) for f in frames]
    
    x,y,w,h = best_box

    tracker = cv2.legacy.TrackerCSRT_create()
    tracker.init(frames[best_frame_idx], (x,y,w,h))

    for frame in frames:
        h_frame, w_frame = frame.shape[:2]
        success, box = tracker.update(frame)

        if success:
            x,y,w,h = [int(v) for v in box]
            x = max(0, x)
            y = max(0, y)
            w = max(1, min(w, w_frame - x))
            h = max(1, min(h, h_frame - y))
            
            face = frame[y: y + h, x: x + w]

            if face is None or face.size == 0:
                face = cv2.resize(frame, (224, 224))
            else:
                face = cv2.resize(face, (224, 224))
            faces.append(face)
        else:
            frame = cv2.resize(frame, (224, 224))
            faces.append(frame)

    return faces


# applying frequency transform
def fft_transform(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return magnitude.astype(np.uint8)