import gradio as gr
from fastai.vision.all import *
import torch


learn = load_learner('gtsrb_model.pkl')

THRESHOLD = 0.91

def classify(img):
    pred, pred_idx, probs = learn.predict(img)
    confidence = probs[pred_idx].item()

    if confidence < THRESHOLD:
        return f"❌ Unknown / Not a traffic sign (Confidence: {confidence:.2f})"

    return f"✅ {pred} ({confidence*100:.2f}%)"


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Textbox(label="Prediction"),
)

demo.launch(share=True)