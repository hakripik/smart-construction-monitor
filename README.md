# Construction Site Safety Detection

This project uses computer vision to detect construction workers, equipment, and safety-related objects in a video. It processes a construction site video frame by frame, sends selected frames to a Roboflow object detection model, and draws bounding boxes, class labels, object counts, and safety violation warnings on the output video.

## Features

- Detects construction workers and equipment
- Identifies safety-related classes such as hard hats and safety vests
- Highlights safety violations such as:
  - Hard Hat OFF
  - Safety Vest OFF
- Draws bounding boxes and confidence scores on the video
- Displays a live object count panel
- Saves the processed result as output.mp4

## Technologies Used

- Python
- OpenCV
- Roboflow API
- Requests
- Base64 encoding

## Requirements

Install the required Python libraries:

bash pip install opencv-python requests 

## Setup

Create a Roboflow API key and set it as an environment variable.

For macOS/Linux:

bash export ROBOFLOW_API_KEY="your_api_key_here" 

For Windows PowerShell:

bash setx ROBOFLOW_API_KEY "your_api_key_here" 

## How to Run

Place your input video in the project folder and name it:

bash construction.mp4 

Then run:

bash python main.py 

The processed video will be saved as:

bash output.mp4 

## How It Works

1. The video is opened using OpenCV.
2. Every third frame is sent to the Roboflow detection API.
3. The latest detection results are reused for the frames in between to reduce API calls.
4. Bounding boxes, labels, confidence scores, and object counts are drawn onto the video.
5. If safety violations are detected, a red warning banner is shown.
6. The final annotated video is exported as output.mp4.

## Notes

- This project requires an active internet connection because it uses the Roboflow API.
- The Roboflow API key should not be uploaded to GitHub.
- The .env file or any file containing secret keys should be added to .gitignore.
- Detection accuracy depends on the Roboflow model, video quality, and camera angle.

## Future Improvements

- Add a web interface using Streamlit or Flask
- Export violation timestamps to a CSV file
- Improve frame skipping for faster processing
- Add support for live webcam detection
- Track objects across frames
