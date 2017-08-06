import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from yolo_pipeline import *

def pipeline_yolo(img):
    output = vehicle_detection_yolo(img)
    return output

if __name__ == "__main__":
    filename = 'examples/test9.jpg'
    image = mpimg.imread(filename)
    yolo_result = pipeline_yolo(image)
    plt.figure()
    plt.imshow(yolo_result)
    plt.title('yolo pipeline', fontsize=30)
    plt.show()
