import os, shutil, pandas as pd, cv2, numpy as np
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm
import time

# ── Config 
model_yolo  = YOLO("Models/besty.pt")
model_yolo.to("cuda")

YOLO_BATCH  = 32
MAX_WORKERS = 8
base_dir    = r"C:\Users\salah\OneDrive\Documents\BDD_Roboflow_Tazi"
output_dir  = r"C:\Users\salah\OneDrive\Documents\BDD_Classifiee"
dossiers    = ["train", "valid", "test"]
CONF_SEUIL  = 0.44
DOSSIER_NC  = "_non_classe"

def _copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)

def _process_multiclass_batch(batch, chemin_dest, stats):
    srcs        = [item[0] for item in batch]
    classes_list = [item[1] for item in batch]
    imgs        = [cv2.imread(s) for s in srcs]
    all_results = model_yolo(imgs, verbose=False, stream=False)

    for img, src, classes_presentes, result in zip(imgs, srcs, classes_list, all_results):
        nom_image      = Path(src).name
        classes_triees = sorted(classes_presentes)
        detections     = []

        for box in result.boxes:
            conf = float(box.conf[0])
            if conf >= CONF_SEUIL:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if (x2 - x1) > 0 and (y2 - y1) > 0:
                    detections.append((img[y1:y2, x1:x2], x1))

        detections.sort(key=lambda d: d[1])

        if len(detections) == len(classes_presentes):
            for (crop, _), classe in zip(detections, classes_triees):
                crop_name = f"{Path(nom_image).stem}_{classe}{Path(nom_image).suffix}"
                cv2.imwrite(os.path.join(chemin_dest, classe, crop_name), crop)
            stats["multi_classe_crop"] += 1
        else:
            tqdm.write(f"  [FALLBACK] {nom_image} — "
                       f"{len(detections)} det / {len(classes_presentes)} classes")
            for classe in classes_presentes:
                _copy(src, os.path.join(chemin_dest, classe, nom_image))
            stats["multi_classe_copie"] += 1

# /
t_global = time.perf_counter()

for dossier in dossiers:
    chemin_source = os.path.join(base_dir, dossier)
    chemin_dest   = os.path.join(output_dir, dossier)
    csv_path      = os.path.join(chemin_source, "_classes.csv")

    if not os.path.exists(csv_path):
        print(f"[AVERT] _classes.csv introuvable dans '{dossier}' — ignoré")
        continue

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    col_fichier = df.columns[0]
    classes     = list(df.columns[1:])

    for nom_classe in classes:
        os.makedirs(os.path.join(chemin_dest, nom_classe), exist_ok=True)
    os.makedirs(os.path.join(chemin_dest, DOSSIER_NC), exist_ok=True)

    stats = {"1_classe": 0, "multi_classe_crop": 0,
             "multi_classe_copie": 0, "aucune_classe": 0, "image_absente": 0}

    # ── Pré-tri 
    filenames  = df[col_fichier].astype(str).str.strip().values
    mask       = df[classes].apply(
                     lambda c: c.astype(str).str.strip() == "1"
                 ).values                          # shape (N, nb_classes)
    n_per_row  = mask.sum(axis=1)                 # nb classes par image
    classes_np = np.array(classes)

    jobs_single, jobs_none, jobs_multi = [], [], []

    for i in range(len(df)):
        nom_image = filenames[i]
        src       = os.path.join(chemin_source, nom_image)

        if not os.path.exists(src):
            stats["image_absente"] += 1
            continue

        n      = int(n_per_row[i])
        present = list(classes_np[mask[i]])        # noms des classes présentes

        if n == 0:
            jobs_none.append(src)
        elif n == 1:
            jobs_single.append((src, present[0]))
        else:
            jobs_multi.append((src, present))

    total = len(jobs_none) + len(jobs_single) + len(jobs_multi)
    print(f"\n[{dossier.upper()}] {total} images — "
          f"single:{len(jobs_single)}  multi:{len(jobs_multi)}  none:{len(jobs_none)}")

    #  1. Copies parallèles 
    def _do_single(job):
        src, classe = job
        _copy(src, os.path.join(chemin_dest, classe, Path(src).name))
        return "single"

    def _do_none(src):
        _copy(src, os.path.join(chemin_dest, DOSSIER_NC, Path(src).name))
        return "none"

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures  = [pool.submit(_do_single, j) for j in jobs_single]
        futures += [pool.submit(_do_none,   s) for s in jobs_none]
        for f in tqdm(futures, desc=f"  Copies ({dossier})", unit="img"):
            kind = f.result()
            if kind == "single": stats["1_classe"]     += 1
            else:                stats["aucune_classe"] += 1

    # Multi-classes → YOLO batch
    if jobs_multi:
        batches = [jobs_multi[i:i+YOLO_BATCH]
                   for i in range(0, len(jobs_multi), YOLO_BATCH)]
        for batch in tqdm(batches, desc=f"  YOLO batch ({dossier})", unit="batch"):
            _process_multiclass_batch(batch, chemin_dest, stats)

    elapsed = time.perf_counter() - t_global
    print(f"\n{'─'*52}")
    print(f"[{dossier.upper()}] Résumé (cumulé {elapsed:.1f}s) :")
    print(f"   1 classe           : {stats['1_classe']}")
    print(f"   Multi-classe crop  : {stats['multi_classe_crop']}")
    print(f"   Multi-classe copie : {stats['multi_classe_copie']}")
    print(f"  Aucune classe      : {stats['aucune_classe']}  → '{DOSSIER_NC}/'")
    print(f"  Image absente      : {stats['image_absente']}")
    print(f"{'─'*52}\n")

print(f"Terminé en {time.perf_counter()-t_global:.1f}s  —  {output_dir}")