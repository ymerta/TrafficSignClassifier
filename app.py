import gradio as gr
from fastai.vision.all import *
import torch
import torch.nn.functional as F
# Etiket ID'lerini anlamlı isimlere çeviren sözlük
label_map = {
    "0": "Speed limit (20km/h)",
    "1": "Speed limit (30km/h)",
    "2": "Speed limit (50km/h)",
    "3": "Speed limit (60km/h)",
    "4": "Speed limit (70km/h)",
    "5": "Speed limit (80km/h)",
    "6": "End of speed limit (80km/h)",
    "7": "Speed limit (100km/h)",
    "8": "Speed limit (120km/h)",
    "9": "No passing",
    "10": "No passing for vehicles > 3.5 tons",
    "11": "Right-of-way at the next intersection",
    "12": "Priority road",
    "13": "Yield",
    "14": "Stop",
    "15": "No vehicles",
    "16": "Vehicles > 3.5 tons prohibited",
    "17": "No entry",
    "18": "General caution",
    "19": "Dangerous curve left",
    "20": "Dangerous curve right",
    "21": "Double curve",
    "22": "Bumpy road",
    "23": "Slippery road",
    "24": "Road narrows on the right",
    "25": "Road work",
    "26": "Traffic signals",
    "27": "Pedestrians",
    "28": "Children crossing",
    "29": "Bicycles crossing",
    "30": "Beware of ice/snow",
    "31": "Wild animals crossing",
    "32": "End of all speed and passing limits",
    "33": "Turn right ahead",
    "34": "Turn left ahead",
    "35": "Ahead only",
    "36": "Go straight or right",
    "37": "Go straight or left",
    "38": "Keep right",
    "39": "Keep left",
    "40": "Roundabout mandatory",
    "41": "End of no passing",
    "42": "End of no passing by vehicles > 3.5 tons"
}

# Modeli yükle
learn = load_learner('gtsrb_model.pkl')



def entropy(probs):
    return -(probs * torch.log(probs + 1e-8)).sum().item()

def classify(img, threshold):
    pred, pred_idx, probs = learn.predict(img)
    confidence = probs[pred_idx].item()
    e = entropy(probs)

    # Hem düşük güven hem yüksek entropy varsa: saçma bir şey olabilir
    if confidence < threshold or e > 2.5:
        return f"❌ Unknown / Not a traffic sign (Conf: {confidence:.2f}, Entropy: {e:.2f})"

    label_name = label_map[str(pred)]
    return f"✅ {label_name} ({confidence*100:.2f}%, Entropy: {e:.2f})"

demo = gr.Interface(
    fn=classify,
    inputs=[gr.Image(type="pil"), gr.Slider(0.0, 1.0, value=0.6, label="Confidence threshold")],
    outputs="text"
)

demo.launch(share=True)