**Emotion Recognition Using CLIP + MLP Adapter**

This project builds a deep-learning based emotion recognition system using the CLIP ViT-B/32 visual encoder combined with a custom multi-layer adapter network.
The goal is to classify facial expressions into seven emotion categories:

😡 Angry

🤢 Disgusted

😨 Fearful

😀 Happy

😐 Neutral

😢 Sad

😮 Surprised

The project is trained and evaluated on the FER2013 emotion dataset.

**🚀 Motivation**

Traditional CNN models struggle to capture global facial relationships such as:

Eye–mouth interaction

Symmetry vs asymmetry

Fine-grained micro-expressions

CLIP’s Vision Transformer (ViT) provides rich semantic image embeddings, but it is not tuned for emotional cues.
This project introduces an Adapter module to specialize CLIP’s embeddings for emotion recognition.

🧠** Architecture Overview**
✔ 1. CLIP (Vision Transformer) Encoder

Pretrained CLIP ViT-B/32

Extracts a 512-dimensional embedding for each face image

Frozen initially, partially unfrozen later (epoch 10)

✔ **2. Emotion Adapter (MLP)**

A deep projection network that learns emotion-specific mapping:

Linear(512 → 1024)
BatchNorm
ReLU
Dropout

Linear(1024 → 512)
BatchNorm
ReLU
Dropout

Linear(512 → 256)
BatchNorm
ReLU
Dropout

Linear(256 → 7)


Outputs logits for 7 emotion classes.

✔ 3. Softmax

Converts logits → probabilities.

🏋️ Training Strategy
🔹 Dataset

FER2013

48×48 grayscale images → resized & normalized for CLIP

🔹 Optimization

AdamW optimizer

Separate LR for CLIP & Adapter

Weight decay for regularization

🔹 Scheduler

Cosine annealing LR schedule
	
🔹 Loss

CrossEntropy Loss

Label smoothing (0.1)

🔹 Mixed Precision

AMP autocast

GradScaler (CUDA)

📊 Evaluation Metrics

The model is evaluated using:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

These metrics reveal both class-wise behavior and global performance.

📈 Training & Validation Curves
Training & Validation Loss + Accuracy (20 Epochs)

<img width="1160" height="785" alt="Screenshot 2025-11-27 072907" src="https://github.com/user-attachments/assets/bf0a2e4b-83f6-43c0-9ad8-85a4e1146ea2" />

Confusion Matrix:

<img width="873" height="748" alt="Screenshot 2025-11-27 073243" src="https://github.com/user-attachments/assets/79845f95-facd-4511-81cc-314038f01991" />

🧪 Final Test Results
🎯 Test Accuracy: 64.85%

🧾 How to Run
1️⃣ Install dependencies
pip install torch torchvision transformers numpy matplotlib seaborn

2️⃣ Prepare Dataset

Place dataset in:

/data/emotion-detection-fer/train
/data/emotion-detection-fer/test

3️⃣ Train the model
python train.py

4️⃣ Evaluate
python evaluate.py

📌 Future Improvements

Add data augmentation

Replace ViT-B/32 with ViT-L/14

Use LoRA/AdapterFusion for more efficient finetuning

Train on RAF-DB or AffectNet

Multimodal extension with Wav2Vec2.0 audio model

🏁 Conclusion

This project demonstrates how a powerful general-purpose model like CLIP can be adapted for fine-grained facial emotion recognition using a custom MLP adapter. Despite being trained on generic image-text pairs, CLIP embeddings prove highly effective when paired with a specialized classifier.

