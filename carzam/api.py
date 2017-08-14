from flask import Flask
import single_prediction as pred

batches, model_vgg19, model_inception_v1 = get_loaded_models_and_batches(img_rows, img_cols, channel, batch_size, data_path, model_path, imagenet_model_path)

img_rows = 224
img_cols = 224
channel = 3
batch_size = 64
data_path = "../data/compcars/data/image/"
model_path = "../models/"
imagenet_model_path = "../imagenet_models/"

app = Flask(__name__)

@app.route("/")
def root():
    img = get_image(img_rows, img_cols, "mini.png")
    classes, classes_top_5 = perform_combined_prediction(model_vgg19, model_inception_v1, batches, img)
    print classes
    print classes_top_5

if __name__ == "__main__":
    app.run()
