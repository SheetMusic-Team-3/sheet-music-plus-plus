import io
import json
import numpy as np
from PIL import Image
import tensorflow as tf
import requests

def sparse_tensor_to_strs(sparse_tensor):
    indices = sparse_tensor[0][0]
    values = sparse_tensor[0][1]
    dense_shape = sparse_tensor[0][2]

    strs = [ [] for i in range(dense_shape[0]) ]

    string = []
    ptr = 0
    b = 0

    for i in range(len(indices)):
        if indices[i][0] != b:
            strs[b] = string
            string = []
            b = indices[i][0]

        string.append(values[ptr])

        ptr = ptr + 1

    strs[b] = string

    return strs

def _process_input(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API
    Args:
        data (obj): the request data, in format of dict or string
        context (Context): an object containing request and configuration details
    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """
    if context.request_content_type == 'application/json':
        data = data.read().decode("utf-8")
        return data if len(data) else ''    
    raise ValueError('{{"error": "unsupported content type {}"}}'.format(
            context.request_content_type or "unknown"))

def _process_output(data, context):
    if data.status_code != 200:
        raise ValueError(data.content.decode('utf-8'))
    response_content_type = context.accept_header
    response = data.content.decode("utf-8")
    sess = tf.Session()
    data = json.loads(response)
    list_tensor = data["outputs"]["fully_connected/BiasAdd"]
    output_tensor = tf.convert_to_tensor(list_tensor, tf.float32)
    decoded, _ = tf.nn.ctc_greedy_decoder(output_tensor, [54])
    sparse_tensor_value = [decoded[0].eval(session=sess)]
    str_predictions = sparse_tensor_to_strs(sparse_tensor_value)
    bytes_pred = str(str_predictions).encode("utf-8")
    return bytes_pred, response_content_type


def handler(data, context):
    """Post-process TensorFlow Serving output before it is returned to the client.
    Args:
        data (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, response content type
    """
    processed_input = _process_input(data, context)
    response = requests.post(context.rest_uri, data=processed_input)
    return _process_output(response, context)


def _return_error(code, message):
    raise ValueError('Error: {}, {}'.format(str(code), message))