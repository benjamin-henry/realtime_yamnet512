import tensorflow as tf
import numpy as np
import csv
import io

def class_names_from_csv(class_map_csv_text):
  """Returns list of class names corresponding to score vector."""
  class_map_csv = io.StringIO(class_map_csv_text)
  class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
  class_names = class_names[1:]  # Skip CSV header
  return class_names


interpreter = tf.lite.Interpreter('./models/yamnet.tflite')

input_details = interpreter.get_input_details()
waveform_input_index = input_details[0]['index']
output_details = interpreter.get_output_details()
scores_output_index = output_details[0]['index']
embeddings_output_index = output_details[1]['index']
spectrogram_output_index = output_details[2]['index']

# Input: .975 seconds of silence as mono 16 kHz waveform samples.
waveform = np.zeros(15600, dtype=np.float32)

interpreter.resize_tensor_input(waveform_input_index, [len(waveform)], strict=True)
interpreter.allocate_tensors()
interpreter.set_tensor(waveform_input_index, waveform)
interpreter.invoke()

scores, embeddings, spectrogram = (
    interpreter.get_tensor(scores_output_index),
    interpreter.get_tensor(embeddings_output_index),
    interpreter.get_tensor(spectrogram_output_index))

print(scores.shape, embeddings.shape, spectrogram.shape)  # (N, 521) (N, 1024) (M, 64)

class_names = class_names_from_csv(open('./yamnet_class_map.csv').read())
print(class_names[scores.mean(axis=0).argmax()])  # Should print 'Silence'.