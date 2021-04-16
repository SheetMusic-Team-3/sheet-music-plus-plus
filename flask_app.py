from flask import Flask, flash, request, send_from_directory
from flask import render_template, redirect, send_file
import os
import tensorflow as tf
from tensorflow.python.framework import ops
import tensorflow.compat.v1 as tfc
import cv2
import numpy as np
from scripts import ctc_utils
from werkzeug.utils import secure_filename
from scripts import sheet_music_parser
import boto3
import json
import env
import base64
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
AWS_KEY = env.AWS_KEY
AWS_SECRET = env.AWS_SECRET

client = boto3.client('sagemaker-runtime', region_name='us-east-1', aws_access_key_id=AWS_KEY, aws_secret_access_key=AWS_SECRET)

UPLOAD_FOLDER = '/home/hilnels/mysite/.uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
PATH = ""

FILENAME = ""

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def yolo_endpoint_pred(client, endpoint_name, content_type, body):
    """ inputs: client -- boto3 sagemaker-runtime client
                endpoint_name -- AWS endpoint name (string)
                content_type -- image/jpeg for YOLO (string)
                body -- request body (image bytes)
        outputs: a list of dictionaries representing each label
    """
    if content_type != "image/jpeg":
        return "Wrong content type"

    ioc_response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=body,
        ContentType=content_type)

    response = ioc_response["Body"].read()
    preds = json.loads(response)
    return preds

def split_to_lines(file_name, preds):
    """ splits a single image of sheet music into multiple images of each line
        inputs: file_name -- local path to image (string)
                preds -- list of dictionaries with the keys: x, y, width, height, conf
        outputs: a list of cv2 images
    """
    pil_img = Image.open(file_name)
    images = []
    width, height = pil_img.size
    for pred in preds:
        if pred['conf'] > 0.7:
            left = 0
            upper = (pred['y'] - pred['height'] / 2) * height
            right = width
            lower = (pred['y'] + pred['height'] / 2) * height
            cropped_im = pil_img.crop((left, upper, right, lower))
            imcv = np.asarray(cropped_im.convert('L'))
            images.append(imcv)
    return images

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Does this have a purpose???
@app.route('/img/<filename>')
def send_img(filename):
	return send_from_directory('', filename)


@app.route('/')
def root():
    return render_template('index.html')


@app.route('/confirm', methods=['GET', 'POST'])
def confirm():
    os.chdir(r'/home/hilnels')

    currdir = os.getcwd()
    print("predict")
    print(currdir)

    if request.method == 'POST':
        print('AWSKEY ' + AWS_KEY)
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            global FILENAME
            FILENAME = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(filename)

    return render_template('confirm.html')

@app.route('/result', methods=['GET', 'POST'])
def predict():
    filename = FILENAME

    os.chdir(r'/home/hilnels')

    currdir = os.getcwd()
    print("predict")
    print(currdir)

    if request.method == 'POST':
        """
        print("AWSKEY " + AWS_KEY)
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("before line 136?")
            print(filename)
            # return redirect(url_for('predict', filename=filename))
        """


        voc_path = "/home/hilnels/mysite/scripts/vocabulary_semantic.txt"

        # Read the dictionary
        dict_file = open(voc_path, 'r')
        dict_list = dict_file.read().splitlines()
        int2word = dict()
        for word in dict_list:
            word_idx = len(int2word)
            int2word[word_idx] = word
        dict_file.close()

        # Restore weights
        width_reduction = 16

        file_path = "/home/hilnels/mysite/.uploads/" + filename

        # Read in file_path
        raw_im = cv2.imread(file_path)
        retval, buffer = cv2.imencode('.jpg', raw_im)
        bytes_jpg = base64.b64encode(buffer)

        preds = yolo_endpoint_pred(client, 'yolov5', 'image/jpeg', bytes_jpg)

        preds = sorted(preds, key=lambda k: k['y'])

        split_images = split_to_lines(file_path, preds)

        if not split_images:
            print("NO PREDICTIONS...")

        # output string
        predict_to_parse = ""

        for image in split_images:
            image = ctc_utils.resize(image, 128)
            image = ctc_utils.normalize(image)
            image = np.asarray(image).reshape(1,image.shape[0],image.shape[1],1)
            print(image.shape)

            seq_lengths = [ image.shape[2] // width_reduction ]

            body =  json.dumps({"signature_name": "predict", "inputs": {"input": image.tolist(), "seq_len": seq_lengths, "rnn_keep_prob": 1.0}})

            ioc_predictor_endpoint_name = 'semantic'
            content_type = 'application/json'
            ioc_response = client.invoke_endpoint(
                EndpointName=ioc_predictor_endpoint_name,
                Body=body,
                ContentType=content_type
             )

            response = ioc_response["Body"].read()

            data = json.loads(response)

            list_tensor = data["outputs"]["fully_connected/BiasAdd"]

            output = tf.convert_to_tensor(list_tensor, tf.float32)

            decoded, _ = tf.nn.ctc_greedy_decoder(output, seq_lengths)

            result = decoded[0].values
            str_predictions = result.numpy().tolist()
            print(str_predictions)
            for w in str_predictions:
                predict_to_parse += int2word[w]
                predict_to_parse += "\n"
                print(int2word[w])

        print(predict_to_parse)

        lilyPond = \
            sheet_music_parser.generate_music(predict_to_parse, filename)

        os.chdir(r'/home/hilnels/mysite/.downloads')
        lilyFile = filename + ".ly"
        with open(lilyFile, "w") as fo:
            fo.write(lilyPond)

        global PATH
        PATH = '/home/hilnels/mysite/.downloads/' + lilyFile

    return render_template('result.html')


@app.route("/download")
def download():

    # os.chdir(r'/home/hilnels/mysite/.downloads')

    currdir = os.getcwd()
    print('download')
    print(currdir)

    global PATH
    return send_file(PATH, as_attachment=True)

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/help")
def help():
    return render_template('help.html')


if __name__ == '__main__':
    app.run()


"""
@app.route('/')
@app.route('/index')
def index():
    return '[testing]'
"""
