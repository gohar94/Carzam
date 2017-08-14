from flask import Flask
import single_prediction as pred

app = Flask(__name__)

@app.route("/")
def root():
    pred.run("/Users/goharirfan/Desktop/carzam/test_images/mini.png")
    return "Hello"

if __name__ == "__main__":
    app.run()
