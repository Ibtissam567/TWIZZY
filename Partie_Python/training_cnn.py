
import os, json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator


DATA_DIR  = r"C:\Users\salah\OneDrive\Bureau\Traffic-Sign-Recognition-master\Traffic-Sign-Recognition-master\Datasets\FTSC_Dataset\classes_tree"

MODEL_DIR       = "Models"
PLOTS_DIR       = "Plots"
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "cnn_panneaux_france.keras")
PLOT_SAVE_PATH  = os.path.join(PLOTS_DIR, "training_curves_france.png")

IMG_SIZE    = (64, 64)
BATCH_SIZE  = 64       
EPOCHS      = 30
LR          = 1e-3
DROPOUT     = 0.4
VALID_SPLIT = 0.2     
# ══════════════════════════════════════════════════════════════════════════════

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Générateurs ───────────────────────────────────────────────────────────────
datagen = ImageDataGenerator(
    rescale            = 1./255,
    validation_split   = VALID_SPLIT,
    rotation_range     = 15,
    width_shift_range  = 0.1,
    height_shift_range = 0.1,
    shear_range        = 0.1,
    zoom_range         = 0.15,
    brightness_range   = [0.7, 1.3],
    horizontal_flip    = False,
    fill_mode          = "nearest"
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size = IMG_SIZE,
    batch_size  = BATCH_SIZE,
    class_mode  = "categorical",
    subset      = "training",    # 80 %
    shuffle     = True,
    seed        = 42
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size = IMG_SIZE,
    batch_size  = BATCH_SIZE,
    class_mode  = "categorical",
    subset      = "validation",  # 20 %
    shuffle     = False,
    seed        = 42
)

NUM_CLASSES = train_gen.num_classes
print(f"\n🎯 Nombre de classes détectées automatiquement : {NUM_CLASSES}")
print(f"   Train  : {train_gen.samples} images")
print(f"   Valid  : {val_gen.samples}  images\n")

# Sauvegarde mapping index → code FTSC (ex: "0" → "A2b")
class_map = {str(v): k for k, v in train_gen.class_indices.items()}
with open(os.path.join(MODEL_DIR, "classes.json"), "w", encoding="utf-8") as f:
    json.dump(class_map, f, ensure_ascii=False, indent=2)
print("✅ classes.json sauvegardé\n")

# Architecture CNN
def build_cnn(num_classes: int) -> tf.keras.Model:
    inp = layers.Input(shape=(*IMG_SIZE, 3))

    # Bloc 1
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Bloc 2
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Bloc 3
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.30)(x)

    # Tête de classification
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(DROPOUT)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)   # adapté automatiquement

    return models.Model(inp, out, name="CNN_Panneaux")

model = build_cnn(NUM_CLASSES)
model.summary()

#  Compilation + Callbacks 
model.compile(
    optimizer = tf.keras.optimizers.Adam(LR),
    loss      = "categorical_crossentropy",
    metrics   = ["accuracy"]
)

cb_list = [
    callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor="val_accuracy", save_best_only=True, verbose=1
    ),
    callbacks.EarlyStopping(
        monitor="val_accuracy", patience=8,
        restore_best_weights=True, verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=4, min_lr=1e-6, verbose=1
    )
]

# Entraînement 
print("\n Début de l'entraînement...\n")
history = model.fit(
    train_gen,
    validation_data     = val_gen,
    epochs              = EPOCHS,
    callbacks           = cb_list,
    workers             = 4,       # 4 threads pour le chargement des images
    use_multiprocessing = False    # Threading au lieu de multiprocessing → stable sous Windows
)

# ── Évaluation finale (sur valid, Test GTSRB non utilisé ici) ─────────────────
print("\n Évaluation finale sur le jeu de validation...")
loss, acc = model.evaluate(val_gen, verbose=0)
print(f"   Val Loss     : {loss:.4f}")
print(f"   Val Accuracy : {acc*100:.2f}%\n")

# ── Courbes ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history["accuracy"],     label="Train Acc")
axes[0].plot(history.history["val_accuracy"], label="Val Acc")
axes[0].set_title("Accuracy par époque")
axes[0].set_xlabel("Époque")
axes[0].legend(); axes[0].grid(True)

axes[1].plot(history.history["loss"],     label="Train Loss")
axes[1].plot(history.history["val_loss"], label="Val Loss")
axes[1].set_title("Loss par époque")
axes[1].set_xlabel("Époque")
axes[1].legend(); axes[1].grid(True)

plt.tight_layout()
plt.savefig(PLOT_SAVE_PATH, dpi=150)
plt.show()
print(f" Courbes  → {PLOT_SAVE_PATH}")
print(f" Modèle   → {MODEL_SAVE_PATH}")
