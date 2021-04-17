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

# Disables TensorFlow INFO, WARNING, and ERROR messages from being printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Creating global variables
AWS_KEY = env.AWS_KEY
AWS_SECRET = env.AWS_SECRET
UPLOAD_FOLDER = '/home/hilnels/mysite/uploads'
DOWNLOAD_FOLDER = '/home/hilnels/mysite/downloads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
IMAGE_RESIZE_HEIGHT = 128
IMAGE_WIDTH_REDUCTION = 16
CONFIDENCE_THRESHOLD = 0.7
VOCAB_PATH = '/home/hilnels/mysite/scripts/vocabulary_semantic.txt'
FILENAME = ''
DOWNLOAD_FILENAME = ''

# TODO: Comment
client = boto3.client(
    'sagemaker-runtime',
    region_name='us-east-1',
    aws_access_key_id=AWS_KEY,
    aws_secret_access_key=AWS_SECRET
)

# TODO: Comment
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def yolo_endpoint_pred(client, endpoint_name, content_type, body):
    ''' inputs: client -- boto3 sagemaker-runtime client
                endpoint_name -- AWS endpoint name (string)
                content_type -- image/jpeg for YOLO (string)
                body -- request body (image bytes)
        outputs: a list of dictionaries representing each label
    '''
    if content_type != 'image/jpeg':
        return 'Wrong content type'

    ioc_response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=body,
        ContentType=content_type)

    response = ioc_response['Body'].read()
    preds = json.loads(response)
    return preds


def split_to_lines(filename, preds):
    ''' splits a single image of sheet music into multiple images of each line
        inputs: filename -- local path to image (string)
                preds -- list of dictionaries with the keys:
                         x, y, width, height, conf
        outputs: a list of cv2 images
    '''
    pil_img = Image.open(filename)
    images = []
    width, height = pil_img.size
    for pred in preds:
        if pred['conf'] > CONFIDENCE_THRESHOLD:
            left = 0
            upper = (pred['y'] - pred['height'] / 2) * height
            right = width
            lower = (pred['y'] + pred['height'] / 2) * height
            cropped_im = pil_img.crop((left, upper, right, lower))
            imcv = np.asarray(cropped_im.convert('L'))
            images.append(imcv)
    return images


def allowed_file(filename):
    ''' TODO
    '''
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def root():
    ''' returns the main page when the app is loaded
        outputs: rendered index HTML page
    '''
    return render_template('index.html')


@app.route('/confirm', methods=['GET', 'POST'])
def confirm():
    ''' checks that uploaded image is valid,
        returns the confirmation page
        outputs: rendered index HTML page
    '''
    # Resets working directory and logs current location
    os.chdir(r'/home/hilnels')
    print('confirm:', os.getcwd())

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
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], FILENAME))
            print(FILENAME)

        return render_template('confirm.html', filename=FILENAME)


@app.route('/send_img/<filename>')
def send_img(filename):
    ''' returns the uploaded image on the confirmation page
        inputs: filename -- the file name of the uploaded image
        outputs: user uploaded image
    '''
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/result', methods=['GET', 'POST'])
def predict():
    ''' processes the title form input,
        sends the uploaded image to the 2 neural nets for processing,
        calls the parser on the semantic output and creates download files,
        returns the result page with links to download files
        outputs: rendered index HTML page
    '''
    # reset working directory and logs current location
    os.chdir(r'/home/hilnels')
    print('predict:', os.getcwd())

    if request.method == 'POST':
        # check if user wants to upload a new image
        print(request.form)
        if request.form['button'] == 'Upload different image':
            return render_template('index.html')

        title = request.form['title']
        # TODO: check if title is empty/contains invalid characters like . or /
        print(title)

        # read the dictionary
        dict_file = open(VOCAB_PATH, 'r')
        dict_list = dict_file.read().splitlines()
        int2word = dict()
        for word in dict_list:
            word_idx = len(int2word)
            int2word[word_idx] = word
        dict_file.close()

        # restore weights

        file_path = UPLOAD_FOLDER + '/' + FILENAME
        print(file_path)

        # read in file_path
        raw_im = cv2.imread(file_path)
        retval, buffer = cv2.imencode('.jpg', raw_im)
        bytes_jpg = base64.b64encode(buffer)

        preds = yolo_endpoint_pred(client, 'yolov5', 'image/jpeg', bytes_jpg)

        preds = sorted(preds, key=lambda k: k['y'])

        split_images = split_to_lines(file_path, preds)

        if not split_images:
            print('NO PREDICTIONS...')

        # output string
        predict_to_parse = ''

        for image in split_images:
            image = ctc_utils.resize(image, IMAGE_RESIZE_HEIGHT)
            image = ctc_utils.normalize(image)
            image = np.asarray(image).reshape(
                1,
                image.shape[0],
                image.shape[1],
                1
            )
            print(image.shape)

            seq_lengths = [image.shape[2] // IMAGE_WIDTH_REDUCTION]

            body = json.dumps({
                'signature_name': 'predict',
                'inputs': {
                    'input': image.tolist(),
                    'seq_len': seq_lengths,
                    'rnn_keep_prob': 1.0
                }
            })

            ioc_predictor_endpoint_name = 'semantic'
            content_type = 'application/json'
            ioc_response = client.invoke_endpoint(
                EndpointName=ioc_predictor_endpoint_name,
                Body=body,
                ContentType=content_type
            )

            response = ioc_response['Body'].read()

            data = json.loads(response)

            list_tensor = data['outputs']['fully_connected/BiasAdd']

            output = tf.convert_to_tensor(list_tensor, tf.float32)

            decoded, _ = tf.nn.ctc_greedy_decoder(output, seq_lengths)

            result = decoded[0].values
            str_predictions = result.numpy().tolist()
            print(str_predictions)
            for w in str_predictions:
                predict_to_parse += int2word[w]
                predict_to_parse += '\n'
                print(int2word[w])

        print(predict_to_parse)

        lilyPond = \
            sheet_music_parser.generate_music(predict_to_parse, title)

        os.chdir(DOWNLOAD_FOLDER)
        global DOWNLOAD_FILENAME
        DOWNLOAD_FILENAME = title + '.ly'
        with open(DOWNLOAD_FILENAME, 'w') as fo:
            fo.write(lilyPond)

    return render_template('result.html')


@app.route('/download')
def download():
    ''' sends download files to user's local machine
        outputs: output LilyPond file
    '''
    # reset working directory and logs current location
    os.chdir(r'/')
    print('download:', os.getcwd())

    return send_file(
        DOWNLOAD_FOLDER + '/' + DOWNLOAD_FILENAME,
        as_attachment=True
    )


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/help')
def help():
    return render_template('help.html')


if __name__ == '__main__':
    app.run()
