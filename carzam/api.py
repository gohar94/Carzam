from flask import Flask, jsonify, request
from werkzeug import secure_filename
import single_prediction as pred
import utils

img_rows = 224
img_cols = 224
channel = 3
batch_size = 64
data_path = "../data/compcars/data/image/"
model_path = "../models/"
imagenet_model_path = "../imagenet_models/"

batches, model_vgg19, model_inception_v1 = pred.get_loaded_models_and_batches(img_rows, img_cols, channel, batch_size, data_path, model_path, imagenet_model_path)
make_strings = utils.get_pickled_list('make_names.p')
model_strings = utils.get_pickled_list('model_names.p')

app = Flask(__name__)

@app.route("/test")
def test():
    img = pred.get_image(img_rows, img_cols, "mini.png")
    classes, classes_top_5 = pred.perform_combined_prediction(model_vgg19, model_inception_v1, batches, img)
    classes_out, classes_out_top_5 = pred.get_classes_to_output(classes, classes_top_5, make_strings, model_strings)
    print classes_out_top_5
    response = jsonify(classes=classes_out, classes_top_5=classes_out_top_5)
    return response

@app.route("/predict", methods = ['POST'])
def predict():
    f = request.files['file']
    f.save(secure_filename(f.filename))
    print f.filename
    img = pred.get_image(img_rows, img_cols, f.filename)
    classes, classes_top_5 = pred.perform_combined_prediction(model_vgg19, model_inception_v1, batches, img)
    classes_out, classes_out_top_5 = pred.get_classes_to_output(classes, classes_top_5, make_strings, model_strings)
    print classes_out_top_5
    response = jsonify(classes=classes_out, classes_top_5=classes_out_top_5)
    return response


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8888)
