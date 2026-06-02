# Construction Site Safety Monitoring System

## Overview

This project uses computer vision and a Roboflow object detection model to monitor construction site safety from video footage. The system detects workers, construction equipment, safety gear, and potential safety violations, then annotates the video with bounding boxes, confidence scores, object counts, and warning messages.

The processed video is saved as a new output file with all detections visualized.

## Features

- Detects construction workers and equipment
- Identifies safety gear such as:
  - Hard Hats
  - Safety Vests
- Detects safety violations:
  - Hard Hat OFF
  - Safety Vest OFF
- Displays confidence scores for each detection
- Shows real-time object counts
- Generates warning banners for detected violations
- Exports an annotated output video

## Technologies Used

- Python
- OpenCV
- Roboflow API
- Requests Library
- Base64 Encoding

## Project Structure

text project/ │ ├── monitor.py ├── requirements.txt ├── README.md ├── construction.mp4 └── output.mp4 

## Installation

Clone the repository:

bash git clone <repository-url> cd <repository-folder> 

Install dependencies:

bash pip install -r requirements.txt 

## API Setup

Create a Roboflow API key and set it as an environment variable.

### macOS/Linux

bash export ROBOFLOW_API_KEY="your_api_key_here" 

### Windows PowerShell

powershell setx ROBOFLOW_API_KEY "your_api_key_here" 

## Usage

Place the input video in the project directory and name it:

text construction.mp4 

Run the program:

bash python monitor.py 

After processing, the annotated video will be saved as:

text output.mp4 

## How It Works

1. Open the input video using OpenCV.
2. Extract video frames.
3. Send selected frames to the Roboflow detection API.
4. Receive object detection predictions.
5. Draw bounding boxes and labels on the video.
6. Display object statistics and safety violations.
7. Save the processed frames into a new output video.

## Safety Monitoring

The system highlights the following safety issues:

- Workers without hard hats
- Workers without safety vests

Detected violations are displayed in a red warning banner at the bottom of the video.

## Future Improvements

- Real-time webcam monitoring
- Object tracking across frames
- Violation logging to CSV files
- Dashboard for safety analytics
- Local YOLO model deployment to remove API dependency

## Author

Created as a Computer Vision and Construction Safety Monitoring project using Roboflow and OpenCV.
