# Body Tracking with MediaPipe

A Python application that uses MediaPipe and OpenCV to track human body poses in real-time through webcam feed.

## Features

- Real-time body pose detection
- Visual skeleton overlay on webcam feed
- Clean, minimal interface focused on pose tracking

## Requirements

- Python 3.7+
- Webcam
- Required Python packages (see installation)

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd shlok
```

2. Install required packages:
```bash
pip install opencv-python mediapipe numpy
```

## Usage

Run the body tracking application:
```bash
cd python-open-cv-arthlete
python "jumping jack.py"
```

- Press 'q' to quit the application
- The application will display your webcam feed with pose tracking lines

## Project Structure

```
shlok/
├── python-open-cv-arthlete/
│   └── jumping jack.py    # Main body tracking application
└── README.md
```

## Technologies Used

- **MediaPipe**: Google's ML framework for pose detection
- **OpenCV**: Computer vision library for webcam handling and image processing
- **NumPy**: Numerical computing library

## License

This project is open source and available under the MIT License. 