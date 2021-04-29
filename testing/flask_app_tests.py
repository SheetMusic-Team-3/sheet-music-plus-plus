import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from flask_app import *

import numpy as np
import boto3
import env
import cv2
from PIL import Image


# AWS SECRETS
AWS_KEY = env.AWS_KEY
AWS_SECRET = env.AWS_SECRET

##############
# UNIT TESTS #
##############


def test_semantic_input_preprocess():
    """ Test the semantic input preprocess function
    """
    pil_img = Image.open("one-line-test.png")
    image = np.asarray(pil_img.convert('L'))
    assert(image is not None)  # Check test image was read properly
    image_list, seq_lengths = semantic_input_preprocess(image)
    assert(type(image_list) == list)  # Check output type
    assert(image_list is not None)  # Check that the list not None
    assert(type(seq_lengths) == list)
    assert(len(seq_lengths) > 0)  # Check calculating seq lengths
    print("Test semantic_input_preprocess passed!")


def test_semantic_endpoint_pred():
    """ Test the semantic endpoint prediction API call and postprocessing
    """
    pil_img = Image.open("one-line-test.png")
    image = np.asarray(pil_img.convert('L'))
    assert(image is not None)  # Check test image was read properly
    # Use separately tested function to generate test input
    input_image, seq_lengths = semantic_input_preprocess(image)
    # AWS Sagemaker client for API authorization
    client = boto3.client(
        'sagemaker-runtime',
        region_name='us-east-1',
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET
    )
    data = semantic_endpoint_pred(client, "semantic", input_image, seq_lengths)
    assert(data is not None)  # Check response is not empty
    assert(type(data) == dict)  # Check data is correct type
    assert(data['outputs'] is not None)  # Check data has tensor data
    print("Test semantic_endpoint_pred passed!")


def test_parse_tensor_to_vocab_indices():
    """ Test the parsing of the semantic inference dictionary output
    """
    pil_img = Image.open("one-line-test.png")
    image = np.asarray(pil_img.convert('L'))
    assert(image is not None)  # Check test image was read properly
    # Use separately tested function to generate test input
    input_image, seq_lengths = semantic_input_preprocess(image)
    # AWS Sagemaker client for API authorization
    client = boto3.client(
        'sagemaker-runtime',
        region_name='us-east-1',
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET
    )
    # Generate test data from endpoint
    data = semantic_endpoint_pred(client, "semantic", input_image, seq_lengths)
    # Output to test
    vocab_indices = parse_tensor_to_vocab_indices(data, seq_lengths)
    assert(vocab_indices is not None)  # Check that an output exists
    assert(type(vocab_indices) == list)  # Check output is list
    assert(len(vocab_indices) > 0)  # Check that some prediction exists
    assert(type(vocab_indices[0]) == int)  # Check list is of type integer
    print("Test parse_tensor_to_vocab_indices passed!")


def test_yolo_endpoint_pred():
    """ Test yolo endpoint Sagemaker API call function
    """
    raw_im = cv2.imread("test.png")
    assert(raw_im is not None)  # Check test image was read properly
    # Preprocess image to bytes
    retval, buffer = cv2.imencode('.jpg', raw_im)
    bytes_jpg = base64.b64encode(buffer)
    assert(type(bytes_jpg) == bytes)  # Check request body is bytes
    # AWS Sagemaker client for API authorization
    client = boto3.client(
        'sagemaker-runtime',
        region_name='us-east-1',
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET
    )
    preds = yolo_endpoint_pred(client, 'yolov5', 'image/jpeg', bytes_jpg)
    assert(preds is not None)  # Check response body not None
    assert(type(preds) == list)  # Check nonempty response is list
    if(len(preds) > 0):
        # Check type and structure of output dictionary
        assert(type(preds[0]) == dict)
        assert(type(preds[0]['label']) == int)
        assert(type(preds[0]['x']) == float)
        assert(type(preds[0]['y']) == float)
        assert(type(preds[0]['width']) == float)
        assert(type(preds[0]['height']) == float)
        assert(type(preds[0]['conf']) == float)
        if(len(preds) > 1):  # Check if list is sorted
            assert(preds[0]['y'] < preds[1]['y'])
    print("Test yolo_endpoint_pred passed!")


def test_split_to_lines():
    """ Test image splitting from prediction dictionary
    """
    filename = "test.png"  # Filename for input to tested function

    # Format image input to get preds
    raw_im = cv2.imread(filename)
    assert(raw_im is not None)  # Check test image was read properly
    # Preprocess image to bytes
    retval, buffer = cv2.imencode('.jpg', raw_im)
    bytes_jpg = base64.b64encode(buffer)
    assert(type(bytes_jpg) == bytes)  # Check request body is bytes
    # AWS Sagemaker client for API authorization
    client = boto3.client(
        'sagemaker-runtime',
        region_name='us-east-1',
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET
    )
    # Get sample preds from separately tested yolo_endpoint_pred function
    preds = yolo_endpoint_pred(client, 'yolov5', 'image/jpeg', bytes_jpg)
    # Tested output
    split_images = split_to_lines(filename, preds)
    assert(split_images is not None)  # Check output not empty
    assert(type(split_images) == list)
    # If there are lines that are split, check image data type
    if(len(split_images) > 0):
        assert(type(split_images[0]) == np.ndarray)
    print("Test split_to_lines passed!")

# RUN TESTS
test_semantic_input_preprocess()
test_semantic_endpoint_pred()
test_parse_tensor_to_vocab_indices()
test_yolo_endpoint_pred()
test_split_to_lines()
