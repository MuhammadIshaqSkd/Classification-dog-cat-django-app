# ml_models.py

import pickle
import numpy as np
from skimage.transform import resize
from skimage.io import imread

# Load the trained model
def load_model():
    model_path = 'ml_models/svm_cat_dog_classifier.pkl'  # Adjust path as needed
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Predict function
def predict_image(image_path):
    # Load the model
    model = load_model()

    # Load and preprocess the image
    img = imread(image_path)
    img_resized = resize(img, (150, 150, 3))
    flat_image = img_resized.flatten()


    # Predict using the model
    prediction = model.predict([flat_image])
    probabilities = model.predict_proba([flat_image])

    print("prediction", prediction)
    print("probabilities", probabilities)
    # Categories
    categories = ['cats', 'dogs']

    return categories[prediction[0]], probabilities[0]
