import tensorflow as tf
import sounddevice as sd
import numpy as np
import csv
import io

BLOCKSIZE = 15600

def class_names_from_csv(class_map_csv_text):
  """Returns list of class names corresponding to score vector."""
  class_map_csv = io.StringIO(class_map_csv_text)
  class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
  class_names = class_names[1:]  # Skip CSV header
  return class_names

class_names = class_names_from_csv(open('./yamnet_class_map.csv').read())


# load model
interpreter = tf.lite.Interpreter('./models/yamnet.tflite')
input_details = interpreter.get_input_details()
waveform_input_index = input_details[0]['index']
output_details = interpreter.get_output_details()
scores_output_index = output_details[0]['index']
embeddings_output_index = output_details[1]['index']
spectrogram_output_index = output_details[2]['index']

# define a method to run the model on incoming data
def interpreter_inference(sd_waveform):
    waveform = np.array(sd_waveform).reshape(BLOCKSIZE)
    interpreter.resize_tensor_input(waveform_input_index, [len(waveform)], strict=True)
    interpreter.allocate_tensors()
    interpreter.set_tensor(waveform_input_index, waveform)
    interpreter.invoke()

    scores, embeddings, spectrogram = (
        interpreter.get_tensor(scores_output_index),
        interpreter.get_tensor(embeddings_output_index),
        interpreter.get_tensor(spectrogram_output_index))
    
    return scores, embeddings, spectrogram

# define a callback which is called every BLOCKSIZE samples received (15600 here)
def callback(indata, frames, time, status):
    if status:
        print(status)
    scores, _, _ = interpreter_inference(indata)
    best_class = class_names[scores.mean(axis=0).argmax()]
    print(best_class)


try:
    # use soundevice InputStream
    with sd.InputStream(
        device=0, channels=max([1]),
        samplerate=16000, blocksize=BLOCKSIZE, callback=callback):
        
        while True:
            pass

except KeyboardInterrupt:
    exit()
except Exception as e:
    exit(type(e).__name__ + ': ' + str(e))