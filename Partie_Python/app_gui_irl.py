
import os, json, threading, time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
from ultralytics import YOLO

#config
YOLO_MODEL_PATH = "Models/besty.pt"
CNN_MODEL_PATH  = "Models/cnn_panneaux_france.keras"   
CLASSES_PATH    = "Models/classes.json"

CONF_YOLO   = 0.35    # seuil de confiance YOLO
CONF_CNN    = 0.70    # seuil CNN
IMG_SIZE    = (64, 64)  

# Couleurs
COLOR_BOX   = (0, 200, 50)      # vert vif
COLOR_TEXT  = (255, 255, 255)   # blanc
COLOR_BG    = (0, 150, 30)      # fond du label


FTSC_CLASSES_FR = {
    # Panneaux de Danger (A)
    "A1a":   "Virage dangereux a droite",
    "A1b":   "Virage dangereux a gauche",
    "A1c":   "Succession de virages",
    "A2b":   "Dos d ane",
    "A2c":   "Rond point",
    "A3b":   "Chaussee retrecie ",
    "A13a":  "Danger : Endroit fréquente par les enfants" ,
    "A13b":  "Danger : Passage pour pietons",
    "A17":   "Danger : Signaux lumineux (Feux tricolores)",
    "A18":   "Danger : Circulation a double sens",
    "A22":   "Danger : Passage d'animaux sauvages",
    "A23":   "Danger : Debouche de cyclistes",
    "A24":   "Danger : Route glissante",
    "AB1":   "Intersection : Priorite a droite",
    "AB3":   "Intersection : Cedez le passage",
    "AB4":   "STOP",
    "AB6":   "Ralentisseur",
    "AB5":   "Route prioritaire",
    "AB9":   "Passage a niveau",
    "AB25":  "Intersection : Carrefour a sens giratoire",
    "AB26":  "Intersection ou vous etes prioritaire",
    "AK5":   "Travaux en cours",
    "AK14":  "Danger inconnu",
    # Interdictions et Obligations (B)
    "B1":    "Sens interdit",
    "B2a":   "Interdiction de tourner a gauche",
    "B2b":   "Interdiction de tourner a droite",
    "B2c":   "Interdiction de faire demi tour",
    "B3":    "Interdiction de depasser",
    "B6a":   "Stationnement interdit",
    "B6b":   "Arret et stationnement interdits",
    "B6d":   "Stationnement alterne semi-mensuel",
    "B8":    "Interdit transport marchandise",
    "B9a":   "Acces interdit aux pietons",
    "B9b":   "Acces interdit aux cycles",
    "B13":   "Limitation de poids par essieu",
    "B14_10": "Limitation de vitesse (10km/h)",
    "B14_20": "Limitation de vitesse (20km/h)",
    "B14_30": "Limitation de vitesse (30km/h)",
    "B14_40": "Limitation de vitesse (40km/h)",
    "B14_50": "Limitation de vitesse (50km/h)",
    "B14_60": "Limitation de vitesse (60km/h)",
    "B14_70": "Limitation de vitesse (70km/h)",
    "B14_80": "Limitation de vitesse (80km/h)",
    "B14_90": "Limitation de vitesse (90km/h)",
    "B14_100": "Limitation de vitesse (100km/h)",
    "B14_110": "Limitation de vitesse (110km/h)",
    "B14_120": "Limitation de vitesse (120km/h)",
    "B15":   "Priorite à la circulation venant en sens inverse",
    "B19":   "Voie pompiers",
    "B21-1": "Obligation de tourner a droite",
    "B21-2": "Obligation de tourner a gauche",
    "B21a1": "Obligation de contournement par la droite",
    "B21c2": "Obligation de contournement par la gauche",
    "B22a":  "Obligation cycliste",
    "B22b":  "Obligation pieton",
    "B30":   "Zone à vitesse limitee (Zone 30)",
    "B40":   "Fin de zone aux cylistes",
    "B51":   "Fin de zone à vitesse limitee",
    "B52":   "Zone pietonne et 20km",
    # Indications (C)
    "C1a":   "Parking",
    "C12":   "Circulation a sens unique",
    "C13a":  "Impasse / Voie sans issue",
    "C20a":  "Passage pour pietons sureleve",
    "C27":   "Dos d'ane",
    # Autres (D, E, J, K, M)
    "D42":   "Balise de virage",
    "J4":    "Balise d'alignement",
    "J5":    "Balise de bifurcation",
    "K5c":   "Balise de chantier",
    "K8":    "Virage dangereux",
    "KC1":   "Sortie de camion",
    "KD10":  "jointure de chemins",
    "KD22":  "Déviation de circulation",
    "M1":    "Panneau de distance",
    "M3a":   "Panneau directionnel",
    "M4c":   "Sauf Motos",
    "M4f":   "3.5t",
    "M6a":   "arret interdit",
    "M6c":   "Duree limitee (Disque)",
    "M6h":   "Sauf handicape",
    "M8a":   "Sens du danger",
    "M8d":   "Etendue du danger",
    "M9c":   "Ceder le passage",
    "M9d":   "Passage Surleve",
    "M9v":   " Sauf cycliste",
    "M9z":   "Rappel",
}

def get_label_fr(raw_label: str) -> str:
    """
    Convertit le code FTSC (ex: 'A13a', 'B14_50') en texte français lisible.
    Si le code n'est pas dans le dictionnaire, on l'affiche tel quel.
    """
    if raw_label in FTSC_CLASSES_FR:
        return FTSC_CLASSES_FR[raw_label]
    return raw_label  



# CHARGEMENT DES MODÈLES
def load_models():
    """Charge YOLO + CNN + mapping de classes."""
    errors = []
    if not os.path.exists(YOLO_MODEL_PATH):
        errors.append(f"Modèle YOLO introuvable : {YOLO_MODEL_PATH}")
    if not os.path.exists(CNN_MODEL_PATH):
        errors.append(f"Modèle CNN introuvable  : {CNN_MODEL_PATH}\n→ Lancez d'abord training_cnn.py")
    if not os.path.exists(CLASSES_PATH):
        errors.append(f"classes.json introuvable : {CLASSES_PATH}")
    if errors:
        messagebox.showerror("Erreur de chargement", "\n".join(errors))
        return None, None, None

    yolo = YOLO(YOLO_MODEL_PATH)
    cnn  = tf.keras.models.load_model(CNN_MODEL_PATH)
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)   # {"0": "stop", "1": "cedez", ...}
    return yolo, cnn, classes


# INFÉRENCE SUR UNE FRAME
def infer_frame(frame_bgr, yolo_model, cnn_model, classes):
    """
    Détecte les panneaux avec YOLO, classifie chaque crop avec le CNN,
    dessine les bounding-boxes et retourne la frame annotée (BGR).
    """
    results  = yolo_model(frame_bgr, verbose=False, conf=CONF_YOLO)
    detections_info = []

    for r in results:
        for box in r.boxes:
            conf_yolo = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_bgr.shape[1], x2), min(frame_bgr.shape[0], y2)

            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue

            crop = frame_bgr[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_resized = cv2.resize(crop_rgb, IMG_SIZE)
            inp = crop_resized.astype("float32") / 255.0
            inp = np.expand_dims(inp, axis=0)

            probs      = cnn_model.predict(inp, verbose=0)[0]
            class_idx  = int(np.argmax(probs))
            conf_cnn   = float(probs[class_idx])
            raw_label  = classes.get(str(class_idx), str(class_idx))
            label      = get_label_fr(raw_label)

            #  Dessin bounding-box
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), COLOR_BOX, 2)

            if conf_cnn >= CONF_CNN:
                short_label = label if len(label) <= 35 else label[:32] + "…"
                text = f"{short_label}"
            else:
                text = "?"
            
            #la police
            font_scale = 0.28
            thickness = 1

            # la taille du texte
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Calcul pour centrer horizontalement
            box_w = x2 - x1
            text_x = x1 + (box_w - tw) // 2
            
            # Sécurité : empêcher le texte de sortir de l'écran à gauche ou à droite
            frame_w = frame_bgr.shape[1]
            text_x = max(2, min(text_x, frame_w - tw - 4))
            
            # Position Y (au-dessus de la boîte)
            ty = max(th + 5, y1 - 4)
            
            # Dessin du fond vert centré
            cv2.rectangle(frame_bgr, (text_x - 1, ty - th - 2), (text_x + tw + 2, ty ), COLOR_BG, -1)
            
            # Dessin du texte par-dessus
            cv2.putText(frame_bgr, text, (text_x, ty - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_TEXT, thickness, cv2.LINE_AA)
            
            detections_info.append({
                "label": label, "conf_yolo": conf_yolo, "conf_cnn": conf_cnn,
                "bbox": (x1, y1, x2, y2)
            })

    return frame_bgr, detections_info


# APPLICATION TKINTER
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🚦 Reconnaissance de Panneaux Routiers")
        self.configure(bg="#1e1e2e")
        self.resizable(True, True)
        self.geometry("1100x720")

        self.yolo_model = None
        self.cnn_model  = None
        self.classes    = None

        self.btn_image  = None
        self.btn_video  = None
        self.btn_stop   = None
        self.btn_camera = None   

        self._video_running = False
        self._video_thread  = None
        self._cap           = None

        self._build_ui()
        self.after(200, self._load_models_async)

    # UI
    def _build_ui(self):
        # ── Barre du haut ──
        top = tk.Frame(self, bg="#181825", pady=8)
        top.pack(fill="x")

        title_lbl = tk.Label(top, text="🚦 Reconnaissance de Panneaux Routiers",
                             font=("Segoe UI", 16, "bold"),
                             fg="#cdd6f4", bg="#181825")
        title_lbl.pack(side="left", padx=16)

        self.status_lbl = tk.Label(top, text="⏳ Chargement des modèles...",
                                   font=("Segoe UI", 10), fg="#fab387", bg="#181825")
        self.status_lbl.pack(side="right", padx=16)

        # ── Boutons ──
        btn_frame = tk.Frame(self, bg="#1e1e2e", pady=6)
        btn_frame.pack(fill="x", padx=16)

        btn_style = dict(font=("Segoe UI", 10, "bold"), bd=0,
                         relief="flat", padx=14, pady=6, cursor="hand2")

        self.btn_image = tk.Button(btn_frame, text="📷  Ouvrir une IMAGE",
                                   bg="#89b4fa", fg="#1e1e2e",
                                   command=self._open_image, **btn_style)
        self.btn_image.pack(side="left", padx=6)

        self.btn_video = tk.Button(btn_frame, text="🎬  Ouvrir une VIDÉO",
                                   bg="#a6e3a1", fg="#1e1e2e",
                                   command=self._open_video, **btn_style)
        self.btn_video.pack(side="left", padx=6)

        self.btn_stop = tk.Button(btn_frame, text="⏹  Stop",
                                  bg="#f38ba8", fg="#1e1e2e",
                                  command=self._stop_video, state="disabled",
                                  **btn_style)
        self.btn_stop.pack(side="left", padx=6)

        # Séparateur visuel
        tk.Frame(btn_frame, bg="#45475a", width=2, height=32).pack(
            side="left", padx=10, pady=4)

        self.btn_camera = tk.Button(btn_frame, text="📹  Caméra en direct",
                                    bg="#cba6f7", fg="#1e1e2e",
                                    command=self._open_camera, **btn_style)
        self.btn_camera.pack(side="left", padx=6)

        # Sélecteur de caméra
        tk.Label(btn_frame, text="Cam :", font=("Segoe UI", 9),
                 fg="#6c7086", bg="#1e1e2e").pack(side="left", padx=(10, 2))
        self.cam_index = tk.IntVar(value=0)
        tk.Spinbox(btn_frame, from_=0, to=5, width=3,
                   textvariable=self.cam_index,
                   font=("Segoe UI", 9), bg="#313244", fg="#cdd6f4",
                   buttonbackground="#45475a", relief="flat",
                   disabledbackground="#1e1e2e").pack(side="left")

        # Zone principale
        main = tk.Frame(self, bg="#1e1e2e")
        main.pack(fill="both", expand=True, padx=16, pady=(0, 10))

        # Canvas d'affichage
        canvas_frame = tk.Frame(main, bg="#313244", bd=2, relief="sunken")
        canvas_frame.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="#1e1e2e",
                                highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Panneau de droite (détections)
        right = tk.Frame(main, bg="#181825", width=240)
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)

        tk.Label(right, text="📋 Détections", font=("Segoe UI", 11, "bold"),
                 fg="#cdd6f4", bg="#181825").pack(pady=(10, 4))

        self.det_listbox = tk.Listbox(right, bg="#313244", fg="#cdd6f4",
                                      font=("Consolas", 9), bd=0,
                                      selectbackground="#45475a",
                                      highlightthickness=0,
                                      activestyle="none")
        self.det_listbox.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # Barre de progression vidéo
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self, variable=self.progress_var,
                                            maximum=100, mode="determinate")
        self.progress_bar.pack(fill="x", padx=16, pady=(0, 6))

        # FPS label
        self.fps_lbl = tk.Label(self, text="", font=("Consolas", 9),
                                fg="#6c7086", bg="#1e1e2e")
        self.fps_lbl.pack()

    # Chargement modèles
    def _load_models_async(self):
        def worker():
            y, c, cls = load_models()
            self.yolo_model = y
            self.cnn_model  = c
            self.classes    = cls
            if y and c and cls:
                self.status_lbl.config(text="✅ Modèles chargés — prêt !",
                                       fg="#a6e3a1")
                self.btn_image.config(state="normal")
                self.btn_video.config(state="normal")
                self.btn_camera.config(state="normal")
            else:
                self.status_lbl.config(text="❌ Erreur de chargement",
                                       fg="#f38ba8")
        self.btn_image.config(state="disabled")
        self.btn_video.config(state="disabled")
        self.btn_camera.config(state="disabled")
        t = threading.Thread(target=worker, daemon=True)
        t.start()

    # Image
    def _open_image(self):
        self._stop_video()
        path = filedialog.askopenfilename(
            title="Choisir une image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Tous", "*.*")]
        )
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Erreur", "Impossible de lire l'image.")
            return
        annotated, dets = infer_frame(frame, self.yolo_model,
                                      self.cnn_model, self.classes)
        self._show_frame(annotated)
        self._update_detections(dets)
        self.status_lbl.config(
            text=f"🖼  {os.path.basename(path)} — {len(dets)} panneau(x) détecté(s)",
            fg="#89dceb")

    # Vidéo 
    #  Caméra en direct
    def _open_camera(self):
        self._stop_video()
        cam_idx = self.cam_index.get()
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)  
        if not cap.isOpened():
            messagebox.showerror(
                "Caméra introuvable",
                f"Impossible d'ouvrir la caméra {cam_idx}.\n"
                "Essaie un autre index (0, 1, 2…) dans le sélecteur."
            )
            return
        # Résolution optimale pour la fluidité
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        self._cap           = cap
        self._video_running = True
        self.btn_stop.config(state="normal")
        self.btn_camera.config(state="disabled")
        self.progress_var.set(0)
        self.status_lbl.config(
            text=f"📹  Caméra {cam_idx} — mode temps réel",
            fg="#cba6f7")

        self._video_thread = threading.Thread(
            target=self._camera_loop, daemon=True)
        self._video_thread.start()

    def _camera_loop(self):
        """Boucle infinie sur le flux webcam — s'arrête sur Stop."""
        t_prev = time.perf_counter()
        frame_idx = 0

        while self._video_running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            annotated, dets = infer_frame(
                frame, self.yolo_model, self.cnn_model, self.classes)
            frame_idx += 1

            now   = time.perf_counter()
            fps   = 1.0 / max(now - t_prev, 1e-6)
            t_prev = now

            def ui_update(af=annotated, d=dets, f=fps, fi=frame_idx):
                self._show_frame(af)
                self._update_detections(d)
                self.fps_lbl.config(
                    text=f"FPS : {f:.1f}  |  Frame {fi}  |  📹 Caméra live")
            self.after(0, ui_update)
            time.sleep(0.001)

        # Fin de boucle
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        self.after(0, lambda: self.btn_stop.config(state="disabled"))
        self.after(0, lambda: self.btn_camera.config(state="normal"))
        self.after(0, lambda: self.status_lbl.config(
            text="⏹ Caméra arrêtée", fg="#fab387"))

    def _open_video(self):
        self._stop_video()
        path = filedialog.askopenfilename(
            title="Choisir une vidéo",
            filetypes=[("Vidéos", "*.mp4 *.avi *.mov *.mkv *.webm"), ("Tous", "*.*")]
        )
        if not path:
            return
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            messagebox.showerror("Erreur", "Impossible d'ouvrir la vidéo.")
            return
        self._video_running = True
        self.btn_stop.config(state="normal")
        total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.status_lbl.config(
            text=f"▶  {os.path.basename(path)} ({total} frames)",
            fg="#a6e3a1")
        self._video_thread = threading.Thread(
            target=self._video_loop, args=(total,), daemon=True)
        self._video_thread.start()

    def _video_loop(self, total_frames):
        frame_idx  = 0
        t_prev     = time.perf_counter()

        while self._video_running:
            ret, frame = self._cap.read()
            if not ret:
                break

            annotated, dets = infer_frame(frame, self.yolo_model,
                                          self.cnn_model, self.classes)
            frame_idx += 1

            # FPS
            now  = time.perf_counter()
            fps  = 1.0 / max(now - t_prev, 1e-6)
            t_prev = now

            # Mise à jour UI (thread-safe via after)
            def ui_update(af=annotated, d=dets, fi=frame_idx, f=fps):
                self._show_frame(af)
                self._update_detections(d)
                pct = (fi / total_frames * 100) if total_frames > 0 else 0
                self.progress_var.set(pct)
                self.fps_lbl.config(text=f"FPS : {f:.1f}  |  Frame {fi}/{total_frames}")
            self.after(0, ui_update)
            time.sleep(0.001)   # laisser l'event loop respirer

        self._cap.release()
        self._video_running = False
        self.after(0, lambda: self.btn_stop.config(state="disabled"))
        self.after(0, lambda: self.status_lbl.config(text="⏹ Vidéo terminée",
                                                      fg="#fab387"))

    def _stop_video(self):
        self._video_running = False
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self.btn_stop.config(state="disabled")
        self.btn_camera.config(state="normal")
        self.progress_var.set(0)
        self.fps_lbl.config(text="")

    # ── Affichage frame
    def _show_frame(self, frame_bgr):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 800, 560

        h, w = frame_bgr.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb).resize((nw, nh), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, anchor="center", image=tk_img)
        self.canvas._tk_img = tk_img   # empêche le garbage-collector

    # ── Panneau de détections
    def _update_detections(self, dets):
        self.det_listbox.delete(0, "end")
        if not dets:
            self.det_listbox.insert("end", "  Aucune détection")
            return

        for i, d in enumerate(dets, 1):
            main_label = d['label']

            # Coupe sur le dernier espace avant 24 chars pour ne pas couper un mot
            if len(main_label) > 24:
                cut = main_label.rfind(" ", 0, 24) or 22
                self.det_listbox.insert("end", f"  #{i} {main_label[:cut]}")
                self.det_listbox.insert("end", f"     {main_label[cut+1:]}")
            else:
                self.det_listbox.insert("end", f"  #{i} {main_label}")

            # Scores de confiance
            self.det_listbox.insert(
                "end",
                f"     YOLO {d['conf_yolo']*100:.0f}% | CNN {d['conf_cnn']*100:.0f}%"
            )
            # Séparateur
            self.det_listbox.insert("end", "  " + "─"*22)

    # ── Fermeture propr
    def on_close(self):
        self._stop_video()
        self.destroy()



if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
