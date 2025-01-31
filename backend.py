# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "flask",
#     "numpy",
#     "opencv-python",
#     "pillow",
#     "flask_cors",
# ]
# ///

import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def apply_edge_detection(image_bytes):
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    _, processed_img_encoded = cv2.imencode('.png', edges)
    return processed_img_encoded.tobytes()


def apply_gaussian_blur(image_bytes, kernel_size=5):
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    _, processed_img = cv2.imencode('.png', blurred)
    return processed_img.tobytes()


def apply_sharpen(image_bytes):
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(img, -1, kernel)
    _, processed_img = cv2.imencode('.png', sharpened)
    return processed_img.tobytes()


def convert_to_gray(image_bytes):
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, processed_img = cv2.imencode('.png', gray)
    return processed_img.tobytes()


def apply_median_blur(image_bytes, kernel_size=3):
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    blurred = cv2.medianBlur(img, kernel_size)
    _, processed_img = cv2.imencode('.png', blurred)
    return processed_img.tobytes()


def apply_sobel_edge(image_bytes):
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = np.uint8(sobel)
    _, processed_img = cv2.imencode('.png', sobel)
    return processed_img.tobytes()


@app.route('/process_image', methods=['POST'])
@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        data = request.get_json()
        image_b64 = data.get('image')
        operation = data.get('operation', 'edge_detection')
        params = data.get('params', {})

        image_bytes = base64.b64decode(image_b64)

        operations = {
            'edge_detection': lambda: apply_edge_detection(image_bytes),
            'gaussian_blur': lambda: apply_gaussian_blur(image_bytes, params.get('kernel_size', 5)),
            'sharpen': lambda: apply_sharpen(image_bytes),
            'grayscale': lambda: convert_to_gray(image_bytes),
            'median_blur': lambda: apply_median_blur(image_bytes, params.get('kernel_size', 3)),
            'sobel_edge': lambda: apply_sobel_edge(image_bytes)
        }

        if operation not in operations:
            return jsonify({"error": f"Operation '{operation}' not supported."}), 400

        processed_image_bytes = operations[operation]()
        processed_image_b64 = base64.b64encode(
            processed_image_bytes).decode('utf-8')

        # Return response with explicit content type
        response = jsonify({
            "processed_image": processed_image_b64,
            "status": "success"
        })
        response.headers['Content-Type'] = 'application/json'
        return response, 200

    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
