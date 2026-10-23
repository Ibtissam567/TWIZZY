import cv2, random, numpy as np
from pathlib import Path
from tqdm import tqdm

# config
INPUT_DIR    = r"C:\Users\salah\OneDrive\Bureau\Traffic-Sign-Recognition-master\Traffic-Sign-Recognition-master\Datasets\FTSC_Dataset\classes_tree\Passage_niveau"   # ← dossier avec tes ~20 images
OUTPUT_DIR   = r"C:\Users\salah\OneDrive\Bureau\Traffic-Sign-Recognition-master\Traffic-Sign-Recognition-master\Datasets\FTSC_Dataset\classes_tree\Passage_niveau_augmente"  # ← sortie
TARGET_COUNT = 600   # nmbre d'images finales
SEED         = 42


random.seed(SEED)
np.random.seed(SEED)

INPUT_DIR  = Path(INPUT_DIR)
OUTPUT_DIR = Path(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# fonctions d'augmentation
def adjust_brightness(img):
    factor = random.uniform(0.4, 1.8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def add_noise(img):
    noise = np.random.normal(0, random.uniform(5, 30), img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def motion_blur(img):
    k = random.choice([3, 5, 7])
    kernel = np.zeros((k, k))
    kernel[k//2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kernel)

def rotate_img(img):
    h, w = img.shape[:2]
    angle = random.uniform(-20, 20)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def zoom_crop(img):
    h, w = img.shape[:2]
    f = random.uniform(0.80, 1.20)
    nh, nw = int(h*f), int(w*f)
    resized = cv2.resize(img, (nw, nh))
    if f >= 1.0:
        y, x = (nh-h)//2, (nw-w)//2
        return resized[y:y+h, x:x+w]
    ph, pw = (h-nh)//2, (w-nw)//2
    return cv2.copyMakeBorder(resized, ph, h-nh-ph, pw, w-nw-pw, cv2.BORDER_REFLECT)

def flip_h(img):
    return cv2.flip(img, 1)

def contrast(img):
    alpha = random.uniform(0.6, 1.8)
    beta  = random.randint(-30, 30)   
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

def perspective(img):
    h, w = img.shape[:2]
    d = random.uniform(0.05, 0.12) * min(h, w)
    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst = np.float32([
        [random.uniform(0,d), random.uniform(0,d)],
        [w-random.uniform(0,d), random.uniform(0,d)],
        [w-random.uniform(0,d), h-random.uniform(0,d)],
        [random.uniform(0,d), h-random.uniform(0,d)]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

AUGMENTS = [adjust_brightness, add_noise, motion_blur,
            rotate_img, zoom_crop, flip_h, contrast, perspective]

def augment_one(img):
    """Applique 1 à 4 transformations aléatoires."""
    for fn in random.sample(AUGMENTS, k=random.randint(1, 4)):
        img = fn(img)
    return img

# Chargement des images sources
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
originals  = [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in extensions]

if not originals:
    raise SystemExit(f" Aucune image trouvée dans {INPUT_DIR}")

print(f"📂 {len(originals)} images sources trouvées")
print(f"🎯 Objectif : {TARGET_COUNT} images\n")

# Copie des originaux 
for p in originals:
    dst = OUTPUT_DIR / p.name
    if not dst.exists():
        import shutil
        shutil.copy2(str(p), str(dst))

#  Augmentation 
needed  = TARGET_COUNT - len(originals)
aug_idx = 0

for i in tqdm(range(needed), desc="Augmentation", unit="img"):
    src_path = random.choice(originals)
    img      = cv2.imread(str(src_path))
    if img is None:
        continue

    aug      = augment_one(img)
    out_name = f"aug_{aug_idx:05d}{src_path.suffix}"
    cv2.imwrite(str(OUTPUT_DIR / out_name), aug)
    aug_idx += 1
