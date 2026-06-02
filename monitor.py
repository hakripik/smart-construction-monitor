import cv2 #pen-source software library used for a wide range of computer vision and machine learning applications
import base64 # built-in module for encoding binary data into ASCII strings and decoding them back into their original binary form
import requests #third-party Python library for making automated HTTP requests and interacting with web services and APIs
import os
API_KEY = os.getenv("ROBOFLOW_API_KEY")
if API_KEY is None:
    raise ValueError("ROBOFLOW_API_KEY environment variable is not set.")
cap = cv2.VideoCapture('construction.mp4')
print('Video opened:', cap.isOpened())
print('Total frames:', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
""

#universe.roboflow.com/ [WORKSPACE] / [PROJECT] / model / [VERSION]
WORKSPACE  = "csr-4br13"
PROJECT    = "construction-equipment-6r96y"
VERSION    = "2"
API_URL    = f"https://detect.roboflow.com/{PROJECT}/{VERSION}"

# Color per class (BGR format)
CLASS_COLORS = {
    "Worker":           (0, 255, 0),
    "person":           (0, 255, 0),
    "Excavator":        (255, 0, 0),
    "Dump Truck":       (255, 100, 0),
    "Tractor Trailer":  (255, 150, 0),
    "Trailer":          (200, 100, 0),
    "Front End Loader": (0, 200, 255),
    "Skid Steer":       (0, 150, 255),
    "Vehicle":          (100, 100, 255),
    "Hard Hat ON":      (0, 255, 200),
    "Hard Hat OFF":     (0, 0, 255),    # RED = violation!
    "Safety Vest ON":   (0, 220, 180),
    "Safety Vest OFF":  (0, 0, 200),    # RED = violation!
    "Ladder":           (255, 255, 0),
}

def detect(frame):
    """Send frame to Roboflow API and get back predictions."""
    _, buffer = cv2.imencode(".jpg", frame) #imencode returns tuple (success_bool, encoded bytes), _ = discarder, buffer=numpy array of bytes representing the jpeg file
    img_b64 = base64.b64encode(buffer).decode("utf-8") #makes binary bytes into Base64 format/text to send to internet
    '''
    # Default quality
    _, buffer = cv2.imencode(".jpg", frame)

    # Higher quality = larger file = more accurate detection
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # Lower quality = smaller file = faster upload = slightly less accurate
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
    '''
    response = requests.post( #sends to roboflow
        API_URL,
        params={"api_key": API_KEY, "confidence": 40, "overlap": 50}, #only when confidence 40+, if overlap more than 50%, merge into one box
        data=img_b64, #img sent in b64 form
        headers={"Content-Type": "application/x-www-form-urlencoded"}, #tell roboflow the format of content so it knows how to read
        timeout=5.0
    )
    response.raise_for_status()
    return response.json().get("predictions", []) #json turns text into python dict. it extracts from the detections list and return [] so no crash.

def draw(frame, predictions):
    """Draw boxes, labels, and safety panel on frame."""
    violations = [] #array
    counts = {} #dictionary

    for pred in predictions: #returned by roboflow with its standard format
        label = pred["class"]
        conf  = pred["confidence"]
        x = int(pred["x"] - pred["width"]  / 2)
        y = int(pred["y"] - pred["height"] / 2)
        w = int(pred["width"])
        h = int(pred["height"])

        color = CLASS_COLORS.get(label, (255, 255, 255)) #(key,defautl value if not found)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{label} {conf:.0%}",
                    (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

        counts[label] = counts.get(label, 0) + 1
        if "OFF" in label:
            violations.append(label)

    # Stats panel (top left)
    panel_h = 30 + len(counts) * 25
    cv2.rectangle(frame, (0, 0), (280, panel_h), (0, 0, 0), -1)
    y_pos = 22
    for cls, count in counts.items():
        color = CLASS_COLORS.get(cls, (255,255,255))
        cv2.putText(frame, f"{cls}: {count}",
                    (8, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2)
        y_pos += 25

    # Safety violation banner
    if violations:
        msg = f"VIOLATION: {', '.join(set(violations))}"
        cv2.rectangle(frame, (0, frame.shape[0]-45), (frame.shape[1], frame.shape[0]), (0,0,180), -1)
        cv2.putText(frame, msg, (10, frame.shape[0]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

    return frame

# ==============================
# 🎬 PROCESS VIDEO
# ==============================
cap = cv2.VideoCapture("construction.mp4")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter("output.mp4",
                      cv2.VideoWriter_fourcc(*"mp4v"),
                      fps, (width, height))

frame_count = 0
print("🚀 Processing video with Roboflow model...")
print("   (this takes longer as each frame is sent to the API)\n")

last_predictions = []  # remember last detections

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Only call API every 3rd frame
    if frame_count % 3 == 0:
        try:
            last_predictions = detect(frame)
        except Exception as e:
            print(f"  Warning: frame {frame_count} skipped ({e})")

    # ✅ Always draw on EVERY frame using last known predictions
    frame = draw(frame, last_predictions)
    out.write(frame)

    if frame_count % 30 == 0:
        print(f"  Frame {frame_count} processed...")

cap.release()
out.release()
print("\n✅ Done! Saved as output.mp4")