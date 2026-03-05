import torch
import cv2
import os
import sys
import random
from datetime import datetime, timedelta
from ultralytics import YOLO
try:
    from utils import (
        detect_color, recognize_plate, detect_pet_breed,
        detect_gender, simulate_audio_event, get_object_color,
        detect_animal_behavior
    )
    from database import salveaza_detectie, init_db
    print("Toate modulele încărcate cu succes!")
except ImportError as e:
    print(f"Eroare import module: {e}")
    print("ișierele utils.py și database.py trebuie să fie în același folder")
    sys.exit(1)

# ===========================================================================
# CONFIGURAȚII
# ===========================================================================
class Config:
    # Căi fișiere
    VIDEO_PATH = r"C:\Users\danie\Desktop\test1.mp4"
    MODELS_DIR = "models"
    
    # Parametri YOLO
    YOLO_MODEL = "yolo11s.pt"
    CONFIDENCE_THRESHOLD = 0.65
    IOU_THRESHOLD = 0.5
    
    # Tracking
    TRACK_PERSIST = True
    MIN_SECONDS_BETWEEN_SAVES = 15
    MIN_CROP_PIXELS = 3000
    
    # Performance
    SKIP_FRAMES = 2
    DISPLAY_SCALE = 1.0
    
    # Culori
    COLOR_MAP = {
        'car': (0, 255, 0),
        'truck': (0, 200, 100),
        'bus': (0, 180, 120),
        'motorcycle': (255, 165, 0),
        'dog': (0, 255, 255),
        'cat': (255, 165, 0),
        'person': (255, 0, 255),
    }
    
    PROCESSED_CLASSES = ['car', 'truck', 'bus', 'motorcycle', 
                         'dog', 'cat', 'person', 'ambulance', 'police']

# ===========================================================================
# CLASA PRINCIPALĂ
# ===========================================================================
class TrafficMonitor:
    def __init__(self):
        self.config = Config()
        self.models = {}
        self.tracking_dict = {}
        self.frame_count = 0
        self.detections_count = 0
        self.start_time = datetime.now()
        self._initialize()
    
    def _initialize(self):
        print("=" * 70)
        print("SISTEM DE MONITORIZARE TRAFIC - INITIALIZARE")
        print("=" * 70)
        self._check_files()
        self._load_models()
        if not init_db(): print(" Continuăm fără baza de date...")
        print("Sistemul este gata!")
        print("=" * 70)
    
    def _check_files(self):
        if not os.path.exists(self.config.VIDEO_PATH):
            print(f"Video nu există: {self.config.VIDEO_PATH}")
            sys.exit(1)
        
        if not os.path.exists(self.config.MODELS_DIR):
            os.makedirs(self.config.MODELS_DIR, exist_ok=True)
        
        yolo_path = os.path.join(self.config.MODELS_DIR, self.config.YOLO_MODEL)
        if not os.path.exists(yolo_path) and not os.path.exists(self.config.YOLO_MODEL):
            print(f"Model YOLO se va descărca...")

    def _load_models(self):
        try:
            yolo_path = os.path.join(self.config.MODELS_DIR, self.config.YOLO_MODEL)
            if os.path.exists(yolo_path):
                self.models['yolo'] = YOLO(yolo_path)
            else:
                self.models['yolo'] = YOLO(self.config.YOLO_MODEL)
        except Exception as e:
            print(f"Eroare YOLO: {e}")
            sys.exit(1)
            
        # Modele opționale
        optional = {'license_plate': 'license_plate.pt'}
        for name, fname in optional.items():
            path = os.path.join(self.config.MODELS_DIR, fname)
            if os.path.exists(path):
                try: self.models[name] = YOLO(path)
                except: pass

    def _should_process_frame(self):
        self.frame_count += 1
        return self.frame_count % self.config.SKIP_FRAMES == 0
    
    def _should_save_detection(self, track_id):
        now = datetime.now()
        if track_id in self.tracking_dict:
            if (now - self.tracking_dict[track_id]).total_seconds() < self.config.MIN_SECONDS_BETWEEN_SAVES:
                return False
        self.tracking_dict[track_id] = now
        return True
    
    # --- LOGICA DE PROCESARE ---
    def _process_vehicle(self, crop, label):
        result = {'culoare': detect_color(crop), 'numar_inmatriculare': None}
        if 'license_plate' in self.models:
            try:
                r = self.models['license_plate'](crop, conf=0.4, verbose=False)[0]
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    plate_text = recognize_plate(crop[y1:y2, x1:x2])
                    if plate_text:
                        result['numar_inmatriculare'] = plate_text
                        break
            except: pass
        if not result['numar_inmatriculare']:
            result['numar_inmatriculare'] = recognize_plate(crop)
        return result
    
    def _process_animal(self, crop, label):
        breed, conf = detect_pet_breed(crop, label)
        if not breed and conf == 0.0: breed = "Necunoscută"
        return {'rasa_animal': breed, 'incredere_rasa': conf, 'comportament': detect_animal_behavior(crop, label)}
    
    def _process_person(self, crop):
        gender, conf = detect_gender(crop)
        return {'gen_persoana': gender, 'incredere_gen': conf}

    # --- SALVARE DB ---
    def _save_to_database(self, label, confidence, track_id, details, frame_num):
        """Salvează detecția dar IGNORĂ coordonatele (x1, y1...) pentru baza de date."""
        detectie_data = {
            'tip_obiect': label,
            'incredere': confidence,
            'track_id': track_id,
            'frame_numar': frame_num,
            'timestamp_original': datetime.now().isoformat(),
            'procesor': 'GPU' if torch.cuda.is_available() else 'CPU'
        }
        detectie_data.update(details)
        saved_id = salveaza_detectie(detectie_data)
        return saved_id

    # --- AFISARE SI LOGARE CU FORMAT: "PERSON 99% | BĂRBAT 60%" ---
    def _draw_detection(self, frame, box, label, details, color, yolo_conf):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        info_lines = []
        
        # 1. Linia Principală: Etichetă YOLO + Procent YOLO
        # Exemplu: "PERSON 99%"
        line1 = f"{label.upper()} {yolo_conf:.0%}"
        
        # Adăugăm a doua parte dacă există atribute secundare
        # Exemplu: " | BĂRBAT 60%"
        if label == 'person' and details.get('gen_persoana'):
            gen_conf = details.get('incredere_gen', 0.0)
            line1 += f" | {details['gen_persoana'].upper()} {gen_conf:.0%}"
            
        elif label in ['dog', 'cat'] and details.get('rasa_animal'):
            rasa_conf = details.get('incredere_rasa', 0.0)
            line1 += f" | {details['rasa_animal'].upper()} {rasa_conf:.0%}"

        info_lines.append(line1)
        
        # Linia 2: Detalii extra (Număr, Culoare)
        extra = []
        if details.get('numar_inmatriculare'): extra.append(f"Nr: {details['numar_inmatriculare']}")
        if details.get('culoare') and details['culoare'] != 'necunoscută': extra.append(details['culoare'])
        
        if extra:
            info_lines.append(" | ".join(extra))
        
        # Desenare linii
        for i, line in enumerate(info_lines):
            y_offset = y1 - 10 - (i * 20)
            if y_offset > 10:
                (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y_offset - th), (x1 + tw, y_offset + 4), color, -1)
                cv2.putText(frame, line, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _log_detection(self, label, yolo_conf, details, saved_id):
        time_str = datetime.now().strftime("%H:%M:%S")
        
        # Iconițe
        icon = "📍"
        if label == 'person': icon = "👤"
        elif label == 'dog': icon = "🐕"
        elif label == 'cat': icon = "🐈"
        elif label in ['car', 'truck', 'bus']: icon = "🚗"
        
        # Construire mesaj bază: [Timp] Icon PERSON 99%
        log_msg = f"[{time_str}] {icon} {label.upper()} {yolo_conf:.0%}"
        
        # Adăugare procent secundar
        if label == 'person' and details.get('gen_persoana'):
            gen_conf = details.get('incredere_gen', 0.0)
            log_msg += f" | {details['gen_persoana']} {gen_conf:.0%}"
            
        elif label in ['dog', 'cat'] and details.get('rasa_animal'):
            rasa_conf = details.get('incredere_rasa', 0.0)
            log_msg += f" | {details['rasa_animal']} {rasa_conf:.0%}"
        
        # Detalii vehicule
        extras = []
        if details.get('numar_inmatriculare'): extras.append(f"Nr: {details['numar_inmatriculare']}")
        if details.get('culoare'): extras.append(f"Cul: {details['culoare']}")
        
        if extras:
            log_msg += " | " + " | ".join(extras)
            
        if saved_id:
            log_msg += f" [ID DB: {saved_id}]"
            
        print(log_msg)

    # --- EXECUȚIE ---
    def run(self):
        print("\nPORNIRE PROCESARE VIDEO...")
        cap = cv2.VideoCapture(self.config.VIDEO_PATH)
        if not cap.isOpened(): return
        
        paused = False
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret: break
                if not self._should_process_frame(): continue
                
                results = self.models['yolo'].track(frame, persist=self.config.TRACK_PERSIST, 
                                                  conf=self.config.CONFIDENCE_THRESHOLD, 
                                                  iou=self.config.IOU_THRESHOLD, verbose=False)[0]
                
                if results.boxes:
                    for box in results.boxes:
                        if box.id is None: continue
                        track_id = int(box.id.item())
                        label = results.names[int(box.cls[0])]
                        conf = box.conf[0].item()
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        if label not in self.config.PROCESSED_CLASSES: continue
                        
                        if not self._should_save_detection(track_id):
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (100,100,100), 1)
                            continue
                        
                        crop = frame[y1:y2, x1:x2]
                        if crop.size < self.config.MIN_CROP_PIXELS: continue
                        
                        details = {'sunet_detectat': simulate_audio_event(label)}
                        if label in ['car', 'truck', 'bus', 'motorcycle', 'ambulance', 'police']:
                            details.update(self._process_vehicle(crop, label))
                        elif label in ['dog', 'cat']:
                            details.update(self._process_animal(crop, label))
                        elif label == 'person':
                            details.update(self._process_person(crop))
                        
                        saved_id = self._save_to_database(label, conf, track_id, details, self.frame_count)
                        color = get_object_color(label, crop)
                        
                        self._draw_detection(frame, (x1, y1, x2, y2), label, details, color, conf)
                        self._log_detection(label, conf, details, saved_id)
                        
                        self.detections_count += 1
            
            cv2.imshow("Monitorizare Trafic", cv2.resize(frame, None, fx=self.config.DISPLAY_SCALE, fy=self.config.DISPLAY_SCALE))
            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            if key == ord('q'): break
            elif key == ord('p'): paused = not paused
        
        cap.release()
        cv2.destroyAllWindows()
        self._show_statistics()

    def _show_statistics(self):
        print("\nProcesare completă!")

if __name__ == "__main__":
    TrafficMonitor().run()