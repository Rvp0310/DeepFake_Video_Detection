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
def extract_frames(vid_path, num_frames = 10, resize=None):
    vid_path = Path(vid_path)

    cap = cv2.VideoCapture(str(vid_path))

    if not cap.isOpened():
        print(f"Error opening video: {vid_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print(f"No frames found in: {vid_path}")
        return

    interval = max(total_frames // num_frames, 1)

    frames = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % interval == 0:

            if resize is not None:
                frame = cv2.resize(frame, resize)

            frames.append(frame)

            if len(frames) >=  num_frames:
                break

        frame_id += 1

    cap.release()
    return frames


# face detection function
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def face_detect_trace(vid_path):
    faces = []

    frames = extract_frames(vid_path)

    first = frames[0]
    gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

    detected = face_detector.detectMultiScale(gray, 1.3, 5)

    if len(detected) == 0:
        frames = [cv2.resize(f, (224, 224)) for f in frames]
        return frames

    x,y,w,h = detected[0]

    tracker = cv2.legacy.TrackerCSRT_create()
    tracker.init(first, (x,y,w,h))

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