from flask import Flask, request # pip install flask
import tensorflow as tf # pip install tensorflow=2.8
from keras.preprocessing.image import img_to_array # pip install tensorflow=2.8
from keras.models import load_model # pip install tensorflow=2.8
import cv2 # pip install opencv-python
import numpy as np # pip install numpy
from flask_cors import CORS
from random import randint
app = Flask(__name__)

model = None
data = dict()
category_to_label = {
    0: "Cardboard",
    1: "Glass Bottle",
    2: "Plastic Bottle",
    3: "Can",
    4: "Food Waste"
}
label_to_category = {
    "Cardboard": 0,
    "Glass Bottle": 1,
    "Plastic Bottle": 2,
    "Can": 3,
    "Food Waste": 4
}

def switch_label(input_label):
    try:
        return category_to_label[input_label]
    except:
        return label_to_category[input_label]

def identify_image(IMG_PATH):
    """
    This is the function where we process the image in tensorflow.
    It should return the label.
    """
    if model is None:
        load_model()
    # random_number = randint(0, len(items)-1)
    # return items[random_number]
    image = cv2.imread(IMG_PATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize_with_padding(image)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    prediction = np.argmax(prediction)
    prediction = switch_label(prediction)
    return prediction

image_size = (100,75)
def resize_with_padding(image):
    new_size = image_size
    height, width, channels = image.shape

    aspect_ratio_orig = float(width) / float(height)
    aspect_ratio_new = float(new_size[0]) / float(new_size[1])

    # Compute the scaling factors for the image
    if aspect_ratio_new > aspect_ratio_orig:
        scale_factor = float(new_size[1]) / float(height)
        new_width = round(scale_factor * width)
        new_height = new_size[1]
    else:
        scale_factor = float(new_size[0]) / float(width)
        new_width = new_size[0]
        new_height = round(scale_factor * height)

    # Resize the image using numpy's resize function
    resized_image = cv2.resize(image, (new_width, new_height))

    # Compute the padding needed to make the new image size match the desired size
    pad_x = (new_size[0] - new_width) // 2
    pad_y = (new_size[1] - new_height) // 2

    # Pad the image with zeros on all sides
    padded_image = np.pad(resized_image, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), 'constant')

    # Fix any rounding point errors
    final_image = cv2.resize(padded_image, new_size)

    return final_image



def load_model():
    global model
    model = tf.keras.models.load_model(".\onboarding_recycle_model.h5")
    return model

# Initializing flask application
app = Flask(__name__)
cors = CORS(app)

@app.route('/')
def main():
    return """
        Application is working
    """


# Process images
@app.route('/image_recognition', methods=['POST'])
def processReq():
    data = request.files["img"]
    data.save("img.jpg")
    resp = identify_image("img.jpg")

    return resp

if __name__ == '__main__':
    app.run(debug=True)
    load_model()