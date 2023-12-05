from uwimg import *
from flask import Flask, send_file, request
from werkzeug.utils import secure_filename
import tempfile 
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
@app.route('/process_image1', methods=['POST'])
def process_images_saturation():
    print('TESTING')
    uploaded_file = request.files['file']

    uploaded_file.save("testing.jpeg")

    # Saturating the swatches image
    im = load_image("testing.jpeg")
    rgb_to_hsv(im)
    shift_image(im, 1, 1)  # Adjust saturation here
    clamp_image(im)
    hsv_to_rgb(im)
    save_image(im, "swatch_saturated")

    return send_file('swatch_saturated.jpg')

@app.route('/process_image2', methods=['POST'])
def process_images_bw():
    print('TESTING')
    uploaded_file = request.files['file']

    uploaded_file.save("testing.jpeg")

    # Black and White filter on skin swatch
    im = load_image("testing.jpeg")
    im = rgb_to_grayscale(im)
    save_image(im, "swatch_bw")

    return send_file('swatch_bw.jpg')


if __name__ == '__main__':
    app.run()
