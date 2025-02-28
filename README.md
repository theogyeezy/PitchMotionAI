# PitchMotionAI
advanced deep learning framework for detecting subtle changes in a pitcher's movement using 240fps high-speed video. The system leverages Optical Flow, 3D Convolutional Networks (3D CNNs), and Vision Transformers (TimeSformer) to analyze mechanics and identify anomalies in pitching motions

🔧 Features
✅ High-Speed Video Processing – Works with 240fps slow-motion footage.
✅ Optical Flow & Skeletal Tracking – Extracts fine-grained motion patterns.
✅ 3D CNN + TimeSformer Architecture – Captures spatial and temporal motion features.
✅ Anomaly Detection – Identifies mechanical inconsistencies or potential injury risks.
✅ Real-time Insights – Can be adapted for live pitch analysis.

🛠️ Setup Instructions

1️⃣ Clone the Repository

git clone https://github.com/yourusername/PitchMotionAI.git
cd PitchMotionAI

2️⃣ Install Dependencies

pip install -r requirements.txt

Key Dependencies
torch – PyTorch for deep learning
torchvision – Pre-trained models and transformations
opencv-python – Video processing & optical flow extraction
einops – Tensor reshaping for Transformers
transformers – TimeSformer model for temporal analysis

📂 Recommended Dataset
🗂️ Data Structure
Prepare video data in the following structure:


/dataset
  /train
    /normal
      - pitch1.mp4
      - pitch2.mp4
    /anomalous
      - bad_pitch1.mp4
      - bad_pitch2.mp4
  /test
    /normal
      - test_pitch1.mp4
    /anomalous
      - test_bad_pitch1.mp4
      
🎥 Where to Get Data?

High-Speed Cameras – 240fps footage from training sessions.
MLB / College Baseball Databases – Access publicly available video archives.
Custom Dataset – Record practice sessions with an iPhone Pro or Sony RX100.

🚀 Model Training

1️⃣ Extract Optical Flow

python preprocess.py --video_dir dataset/train --save_dir processed_data

2️⃣ Train the Model

python train.py --epochs 20 --batch_size 8 --lr 0.0001

3️⃣ Run Inference

python infer.py --video_path test_video.mp4

📊 Example Use Case

After training, you can pass a new high-speed video through the model to analyze a pitcher's mechanics:

from model import PitchingMovementModel
import torch
import cv2

# Load trained model
model = PitchingMovementModel()
model.load_state_dict(torch.load("pitch_motion_model.pth"))
model.eval()

# Process video for analysis
video_path = "test_pitch.mp4"
motion_data = extract_optical_flow(video_path)

# Predict pitch quality
with torch.no_grad():
    prediction = model(torch.tensor(motion_data).unsqueeze(0))
    print("Prediction:", prediction)

📜 License
This project is licensed under the MIT License – free to use, modify, and distribute!
