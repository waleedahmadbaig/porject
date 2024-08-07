import cv2, os
import numpy as np
from tensorflow.keras.models import load_model
import imutils
import base64
from django.conf import settings

############################################################
def crop_brain_contour(image, plot=False):
    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)


    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
    return new_image

############################################################

# def load_data(filename, image_size):
#     """
#     Read images, resize and normalize them.
#     Arguments:
#         dir_list: list of strings representing file directories.
#     Returns:
#         X: A numpy array with shape = (#_examples, image_width, image_height, #_channels)
#         y: A numpy array with shape = (#_examples, 1)
#     """

#     # load all images in a directory
#     X = []
#     image_width, image_height = image_size

#     # load the image
#     image = cv2.imread(filename)
#     # crop the brain and ignore the unnecessary rest part of the image
#     image = crop_brain_contour(image, plot=False)
#     # resize image
#     image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
#     # normalize values
#     image = image / 255.
#     # convert image to numpy array and append it to X
#     X.append(image)

#     X = np.array(X)

#     return X

# ############################################################
# def get_prediction_from_image(imageName):
#     IMG_WIDTH, IMG_HEIGHT = (240, 240)

#     X = load_data(imageName, (IMG_WIDTH, IMG_HEIGHT))
#     best_model = load_model(filepath='cv/model/cnn-parameters-improvement-23-0.91.model')
#     y = best_model.predict(X)
#     return str(y[0][0])

# def get_prediction_from_image(imageName):
#     blank_image = np.zeros((10,10,3), np.uint8)
#     cv2.imwrite("filename23.jpg", blank_image)
#     return "Hey"

def get_jpg_image(imageData):
    ret, frame_buff = cv2.imencode('.jpg', imageData) #could be png, update html as well
    return base64.b64encode(frame_buff)

def get_prediction_from_image_upload(imageData):
        # load all images in a directory
    X = []

    # load the image
    # np_buffer = np.frombuffer(imageData, np.uint8)
    # image = cv2.imdecode(np_buffer, 128 | 1)
    # image = Image.open(io.BytesIO(imageData.read()))
    original_image = cv2.imdecode(np.fromstring(imageData.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    # crop the brain and ignore the unnecessary rest part of the image
    image_cropped = crop_brain_contour(original_image, plot=False)
    # resize image
    image = cv2.resize(image_cropped, dsize=(240, 240), interpolation=cv2.INTER_CUBIC)
    # normalize values
    image = image / 255.
    # convert image to numpy array and append it to X
    X.append(image)

    X = np.array(X)
    best_model = load_model(filepath=os.path.join(settings.BASE_DIR, "cv/model/cnn-parameters-improvement-23-0.91.model"))
    y = best_model.predict(X)

    return y[0][0], str(imageData), get_jpg_image(original_image)

def get_prediction_from_image_upload_new(image_path):
    # load all images in a directory
    X = []

    # load the image
    # np_buffer = np.frombuffer(imageData, np.uint8)
    # image = cv2.imdecode(np_buffer, 128 | 1)
    # image = Image.open(io.BytesIO(imageData.read()))
    original_image = cv2.imread(str(image_path))
    # crop the brain and ignore the unnecessary rest part of the image
    image_cropped = crop_brain_contour(original_image, plot=False)
    # resize image
    image = cv2.resize(image_cropped, dsize=(240, 240), interpolation=cv2.INTER_CUBIC)
    # normalize values
    image = image / 255.
    # convert image to numpy array and append it to X
    X.append(image)

    X = np.array(X)
    best_model = load_model(filepath=os.path.join(settings.BASE_DIR, "cv/model/cnn-parameters-improvement-23-0.91.model"))
    y = best_model.predict(X)

    return y[0][0], str(image_path), get_jpg_image(original_image)
