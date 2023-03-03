from flask import Flask, request,render_template
import pickle
import keras
from werkzeug.utils import secure_filename
import os
import numpy as np

from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)


mapper = {0: "Cyst", 1: "Normal", 2: "Stone", 3: "Tumor"}
basepath = os.path.dirname(__file__)
model = load_model((os.path.join(basepath,'kidney_stone.h5')))
model.make_predict_function()  
# model = pickle.load(open(os.path.join(basepath,'model.pkl'),"rb"))

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(100,100))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds


@app.route('/',methods=['GET'])
def index():
    print(__file__)
    return render_template('index.html')

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        img = request.files["file"]
        # Save the file to ./uploads
        file_path = os.path.join(basepath, "uploads", secure_filename(img.filename))
        img.save(file_path)

        # Make prediction
        preds = model_predict(file_path,model)
        predicted = preds.argmax(axis=-1)
        result = mapper[predicted]
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)