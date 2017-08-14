from flask import Flask, jsonify
import single_prediction as pred

img_rows = 224
img_cols = 224
channel = 3
batch_size = 64
data_path = "../data/compcars/data/image/"
model_path = "../models/"
imagenet_model_path = "../imagenet_models/"

batches, model_vgg19, model_inception_v1 = pred.get_loaded_models_and_batches(img_rows, img_cols, channel, batch_size, data_path, model_path, imagenet_model_path)

app = Flask(__name__)

@app.route("/")
def root():
    img = pred.get_image(img_rows, img_cols, "mini.png")
    classes, classes_top_5 = pred.perform_combined_prediction(model_vgg19, model_inception_v1, batches, img)
    print classes
    print classes_top_5
    response = jsonify(classes=classes, classes_top_5=classes_top_5)
    return response

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8888)
