from flask import Flask, flash, request,send_from_directory,render_template, redirect, url_for, send_file
import os
import tensorflow as tf
from tensorflow.python.framework import ops
import tensorflow.compat.v1 as tfc
import cv2
import numpy as np
from zipfile import ZipFile
from PIL import Image, ImageFont, ImageDraw
import ctc_utils
from werkzeug.utils import secure_filename
import sheet_music_parser


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

UPLOAD_FOLDER = '/home/hilnels/mysite/.uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
PATH = ""

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/img/<filename>')
def send_img(filename):
	return send_from_directory('', filename)

@app.route("/")
def root():
	return render_template('demo-index.html')

@app.route('/predict', methods = ['GET', 'POST'])
def predict():

    os.chdir(r'/home/hilnels')

    currdir = os.getcwd()
    print("predict")
    print(currdir)

    if request.method == 'POST':
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

        voc_path = "/home/hilnels/mysite/vocabulary_semantic.txt"
        model_path = "/home/hilnels/mysite/semantic_model.meta"

        ops.reset_default_graph()
        sess = tfc.InteractiveSession()

        # Read the dictionary
        dict_file = open(voc_path,'r')
        dict_list = dict_file.read().splitlines()
        int2word = dict()
        for word in dict_list:
            word_idx = len(int2word)
            int2word[word_idx] = word
        dict_file.close()

        # Restore weights
        tfc.disable_v2_behavior()
        saver = tfc.train.import_meta_graph("semantic_model.meta")
        saver.restore(sess,"semantic_model.meta"[:-5])

        graph = ops.get_default_graph()

        input = graph.get_tensor_by_name("model_input:0")
        seq_len = graph.get_tensor_by_name("seq_lengths:0")
        print(seq_len)
        rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
        height_tensor = graph.get_tensor_by_name("input_height:0")
        width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
        logits = ops.get_collection("logits")[0]

        # Constants that are saved inside the model itself
        WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])

        decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len)

        file_name = "/home/hilnels/mysite/.uploads/" + filename
        image = cv2.imread(file_name,False)
        image = ctc_utils.resize(image, HEIGHT)
        image = ctc_utils.normalize(image)
        image = np.asarray(image).reshape(1,image.shape[0],image.shape[1],1)
        seq_lengths = [ image.shape[2] / WIDTH_REDUCTION ]
        prediction = sess.run(decoded,
                              feed_dict={
                                  input: image,
                                  seq_len: seq_lengths,
                                  rnn_keep_prob: 1.0
                              })
        print(prediction)

        predict_to_parse = ""

        str_predictions = ctc_utils.sparse_tensor_to_strs(prediction)
        for w in str_predictions[0]:
            predict_to_parse += int2word[w]
            predict_to_parse += "\n"
            print(int2word[w])

        print(predict_to_parse)

        lilyPond = sheet_music_parser.generate_music(predict_to_parse, filename)

        os.chdir(r'/home/hilnels/mysite/.downloads')
        lilyFile = filename + ".ly"
        with open(lilyFile, "w") as fo:
            fo.write(lilyPond)

        global PATH
        PATH = '/home/hilnels/mysite/.downloads/' + lilyFile

    return render_template('demo-result.html')

@app.route("/download")
def download():

    os.chdir(r'/home/hilnels/mysite/.downloads')

    currdir = os.getcwd()
    print("download")
    print(currdir)

    global PATH
    return send_file(PATH, as_attachment=True)


if __name__=="__main__":
	app.run()





"""
@app.route('/')
@app.route('/index')
def index():
	return '[testing]'
"""