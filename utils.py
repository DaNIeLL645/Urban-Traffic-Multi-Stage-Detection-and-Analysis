import cv2
import numpy as np
import easyocr
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import os
import warnings

warnings.filterwarnings('ignore')

print("=" * 70)
print("SISTEM DE MONITORIZARE TRAFIC - MODULE INITIALIZATE")
print("=" * 70)

CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'debug': True,
    'min_confidence_breed': 0.50,  
    'min_confidence_gender': 0.60
}

try:
    use_gpu = torch.cuda.is_available()
    reader = easyocr.Reader(['ro', 'en'], gpu=use_gpu)
    print(f"[UTILS] EasyOCR încărcat! (GPU: {use_gpu})")
except Exception as e:
    print(f"[UTILS] EasyOCR nu s-a putut încărca: {e}")
    reader = None

def detect_color(crop):
    if crop is None or crop.size == 0: return None

    try:
        img_small = cv2.resize(crop, (64, 64))
        
        h, w = img_small.shape[:2]
        start_x, end_x = int(w * 0.35), int(w * 0.65)
        start_y, end_y = int(h * 0.35), int(h * 0.65)
        center = img_small[start_y:end_y, start_x:end_x]
        
        hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        mean_h = np.mean(h)
        mean_s = np.mean(s)
        mean_v = np.mean(v)
        IS_COLOR = mean_s > 35 

        if not IS_COLOR:
            if mean_v < 60:
                return "negru"
            elif mean_v < 130:
                return "gri"      
            elif mean_v < 200:
                return "argintiu" 
            else:
                return "alb"      
        else:
            hue = mean_h
            
            if mean_v < 50: 
                return "negru" 

            if (hue < 10) or (hue > 170):
                return "roșu"
            elif 10 <= hue < 25:
                return "portocaliu"
            elif 25 <= hue < 35:
                return "galben"
            elif 35 <= hue < 85:
                return "verde"
            elif 85 <= hue < 125:
                return "albastru" 
            elif 125 <= hue < 145:
                return "mov"
            elif 145 <= hue < 170:
                return "roz/magenta"
            
            return "gri" 

    except Exception as e:
        return None

def recognize_plate(crop):
    if crop is None or crop.size == 0 or reader is None: 
        return None
        
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        if gray.shape[1] < 300:
            scale = 300 / gray.shape[1]
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        binary = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
        
        result = reader.readtext(binary, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', detail=0)
        
        full_text = "".join(result).replace(' ', '').upper()
        
        if 5 <= len(full_text) <= 8:
            return full_text
            
        return None
        
    except Exception as e:
        return None

def _load_pet_breed_model():
    try:
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        model = models.mobilenet_v2(weights=weights)
        
        device = torch.device(CONFIG['device'])
        model.to(device)
        model.eval()
        
        return {
            'model': model,
            'transform': weights.transforms(),
            'labels': weights.meta["categories"]
        }
    except Exception as e:
        print(f"[UTILS] Eroare MobileNet: {e}")
        return None

def detect_pet_breed(crop, animal_type=None):
    if crop is None or crop.size == 0: return None, 0.0
    
    if not hasattr(detect_pet_breed, 'model_data'):
        detect_pet_breed.model_data = _load_pet_breed_model()
        
    if detect_pet_breed.model_data is None:
        return None, 0.0

    try:
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        transform = detect_pet_breed.model_data['transform']
        input_tensor = transform(pil_img).unsqueeze(0).to(CONFIG['device'])
        
        with torch.no_grad():
            output = detect_pet_breed.model_data['model'](input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
        
        conf, class_idx = torch.max(probabilities, 1)
        conf_val = conf.item()
        
        raw_breed = detect_pet_breed.model_data['labels'][class_idx.item()]
        translated_breed = raw_breed.replace('_', ' ').title()
        
        if conf_val < CONFIG['min_confidence_breed']: 
            return "Necunoscută", conf_val

        if "nautilus" in translated_breed.lower():
            return "Necunoscută", 0.0

        return translated_breed, conf_val

    except Exception as e:
        if CONFIG['debug']: print(f"Eroare rasă: {e}")
        return "Eroare", 0.0

def _load_gender_model():
    proto_path = "models/gender_deploy.prototxt"
    model_path = "models/gender_net.caffemodel"
    
    if not os.path.exists(proto_path) or not os.path.exists(model_path):
        print(f"[UTILS] LIPSĂ MODELE GEN! Verifică folderul 'models'.")
        return None
        
    try:
        net = cv2.dnn.readNet(model_path, proto_path)
        
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("[UTILS]Gen Detection running on CUDA!")
        else:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
        print("[UTILS] Model gen încărcat!")
        return net
    except Exception as e:
        print(f"[UTILS] Eroare critică model gen: {e}")
        return None

def detect_gender(crop):
    if crop is None or crop.size == 0: return None, 0.0
    
    if not hasattr(detect_gender, 'net'):
        detect_gender.net = _load_gender_model()
    
    if detect_gender.net is None:
        return None, 0.0
    
    try:
        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        blob = cv2.dnn.blobFromImage(crop, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        detect_gender.net.setInput(blob)
        preds = detect_gender.net.forward()
        
        male_prob = preds[0][0]
        female_prob = preds[0][1]
        
        if male_prob > female_prob:
            return "Bărbat", float(male_prob)
        else:
            return "Femeie", float(female_prob)
        
    except Exception as e:
        if CONFIG['debug']: print(f"Eroare gen: {e}")
        return None, 0.0

def detect_animal_behavior(crop, animal_type):
    if crop is None or crop.size == 0: return "necunoscut"
    h, w = crop.shape[:2]
    return "stând" if h/w > 1.2 else "culcat"

def simulate_audio_event(object_type):
    return None

def get_object_color(label, crop):
    colors = {
        'car': (0, 255, 0),       
        'dog': (0, 255, 255),     
        'cat': (255, 165, 0),     
        'person': (255, 0, 255),  
        'truck': (0, 165, 255)    
    }
    return colors.get(label, (0, 0, 255)) 

if __name__ == "__main__":
    print("\nVerificare utils.py completă! Sistem gata de rulare.")