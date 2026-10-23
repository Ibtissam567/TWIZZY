

import tensorflow as tf
import tf2onnx

#  Configuration 
MODEL_KERAS_PATH = "Models/cnn_panneaux_france.keras"
MODEL_ONNX_PATH  = "Models/cnn_panneaux_frnace.onnx"

model = tf.keras.models.load_model(MODEL_KERAS_PATH)


spec = (tf.TensorSpec((None, 64, 64, 3), tf.float32, name="input"),)

print(" Conversion en cours vers ONNX")
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

#  Sauvegarde 
with open(MODEL_ONNX_PATH, "wb") as f:
    f.write(model_proto.SerializeToString())

print(f"Le modèle universel est sauvegardé ici : {MODEL_ONNX_PATH}")