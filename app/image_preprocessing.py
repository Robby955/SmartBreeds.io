import tensorflow as tf
from google.cloud import storage
import os
import tempfile
from PIL import Image
from PIL import ImageOps
import numpy as np

print('Loading image crop model...')
def download_model(bucket_name, model_prefix, destination_dir):
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # List all objects in the specified bucket with the prefix
    blobs = bucket.list_blobs(prefix=model_prefix)
    for blob in blobs:
        # Construct the local path
        local_path = os.path.join(destination_dir, blob.name)
        if not os.path.exists(os.path.dirname(local_path)):
            os.makedirs(os.path.dirname(local_path))
        # Download the file
        blob.download_to_filename(local_path)
        print(f"Downloaded {blob.name} to {local_path}")

def load_model(destination_dir):
    model_dir = os.path.join(destination_dir, 'saved_model')
    if not os.path.exists(model_dir):
        raise Exception(f"Model directory {model_dir} not found.")
    return tf.saved_model.load(model_dir)

# Specify your bucket name and model prefix
bucket_name = 'predict-breed-models'
model_prefix = ''  # If the model is in the root of the bucket
destination_dir = 'downloaded_model'

# Download and load the model
download_model(bucket_name, model_prefix, destination_dir)
ssd_model = load_model(destination_dir)

# Now ssd_model is loaded and can be used for inference


def crop_dog(image_path, model):
    # Load and correct the image orientation
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)

    # Resize the image for MobileNet SSD (300x300)
    img_resized = img.resize((300, 300))

    # Convert the image to uint8
    img_array = np.array(img_resized, dtype=np.uint8)

    # Convert the image to a tensor
    image_tensor = tf.convert_to_tensor(img_array)
    image_tensor = tf.expand_dims(image_tensor, axis=0)

    # Perform detection
    detector_output = model(image_tensor)

    # Process the detector's output
    boxes, scores, classes, num = detector_output['detection_boxes'][0], detector_output['detection_scores'][0], \
    detector_output['detection_classes'][0], int(detector_output['num_detections'][0])

    # Filter out dog detections (assuming class 17 is for dogs)
    dog_boxes = boxes[tf.logical_and(classes == 17, scores >= 0.05)]

    if tf.size(dog_boxes) == 0:
        return img, None  # No dog found

    # Use the first box (highest confidence)
    box = dog_boxes[0].numpy()
    ymin, xmin, ymax, xmax = box

    # Crop the original image (not resized) to the bounding box
    cropped_img_array = np.array(img)[int(ymin * img.height):int(ymax * img.height),
                        int(xmin * img.width):int(xmax * img.width), :]
    cropped_img = Image.fromarray(cropped_img_array)

    # Resize the cropped image to 224x224
    cropped_img = cropped_img.resize((224, 224))

    return img, cropped_img


'''
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from google.cloud import storage
import os
import logging

logging.basicConfig(level=logging.INFO)

def download_model(bucket_name, model_prefix, destination_dir):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=model_prefix)

    for blob in blobs:
        # We're adjusting the blob_path to include 'saved_model' subdirectory
        blob_path = 'saved_model/' + blob.name[len(model_prefix):].lstrip('/')
        local_path = os.path.join(destination_dir, blob_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        logging.info(f"Downloaded {blob.name} to {local_path}")

def load_model():
    model_path = "/app/ssd_mobile/saved_model"
    return tf.saved_model.load(model_path)

def crop_dog(image_path, model):
    """Crops the image of a dog from the given image path using the specified model."""
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)

    img_resized = img.resize((300, 300))
    img_array = np.array(img_resized, dtype=np.uint8)
    image_tensor = tf.convert_to_tensor(img_array)
    image_tensor = tf.expand_dims(image_tensor, axis=0)

    detector_output = model(image_tensor)
    boxes, scores, classes, _ = detector_output['detection_boxes'][0], detector_output['detection_scores'][0], detector_output['detection_classes'][0], int(detector_output['num_detections'][0])
    dog_boxes = boxes[tf.logical_and(classes == 17, scores >= 0.0)]

    if tf.size(dog_boxes) == 0:
        return img, None  # No dog found.

    box = dog_boxes[0].numpy()
    ymin, xmin, ymax, xmax = box
    cropped_img_array = np.array(img)[int(ymin * img.height):int(ymax * img.height), int(xmin * img.width):int(xmax * img.width), :]
    cropped_img = Image.fromarray(cropped_img_array).resize((224, 224))

    return img, cropped_img
    '''