import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np
import pandas as pd
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

#load drone model
extractor = AutoFeatureExtractor.from_pretrained("preszzz/drone-audio-detection-05-17-trial-0")
drone_model = AutoModelForAudioClassification.from_pretrained("preszzz/drone-audio-detection-05-17-trial-0")


#load YAMNet(bird) model
bird_model = hub.load("https://tfhub.dev/google/yamnet/1")

#load class labels
class_map_path = tf.keras.utils.get_file(
    "yamnet_class_map.csv",
    "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
)
#convert it to a panadas dataframe
class_map_df = pd.read_csv(class_map_path)

#map classes to Drone/Bird
bird_classes = [
    "Bird",
    "Bird vocalization, bird call, bird song",
    "Chirp, tweet",
    "Owl",
    "Crow",
    "Sparrow",
    "Cuckoo",
    "Parakeet, parrot",
    "Chickadee",
    "Quail",
    "Nightingale",
    "Wren",
    "Blackbird",
    "Woodpecker",
    "Seabird",
    "Lark, meadowlark"
]

def is_bird(filepath):
    waveform, sr = librosa.load(filepath, sr=16000, mono=True)
    scores, _, _ = bird_model(waveform)
    mean_scores = np.mean(scores, axis=0)

    # Normalize so they sum to 1
    mean_scores = mean_scores / np.sum(mean_scores)

    #Map predictions to classes
    predictions = {class_map_df['display_name'][i]: mean_scores[i]
                   for i in range(len(mean_scores))}

    # Group into bird/not_bird
    bird_score = sum([predictions.get(c, 0) for c in bird_classes ])
    other_score = 1 - bird_score

    categories = {"bird": bird_score, "not_bird": other_score}
    prediction = "bird" if bird_score > 0.3 else "not_bird"
    return prediction


def is_drone(filepath):
    audio, sr = librosa.load(filepath, sr=extractor.sampling_rate)
    inputs = extractor(audio, sampling_rate=extractor.sampling_rate, return_tensors="pt")

    #Run inference
    with torch.no_grad():
        outputs = drone_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

    #Get prediction
    predicted_class = probs.argmax(dim=-1).item()
    label = drone_model.config.id2label[predicted_class]
    confidence = probs[0, predicted_class].item()
    return label

def classify_audio(filepath):
    if is_bird(filepath) == 'bird':
        return 'Bird'
    elif is_drone(filepath) == 'drone':
        return 'Drone'
    else:
        return 'Other'

