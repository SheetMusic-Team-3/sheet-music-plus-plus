from flask import Flask, flash, request,send_from_directory,render_template, redirect, url_for
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


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

UPLOAD_FOLDER = '/home/hilnels/mysite/.uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# def sparse_tensor_to_strs(sparse_tensor):
# 	indices = sparse_tensor[0][0]
# 	values = sparse_tensor[0][1]
# 	dense_shape = sparse_tensor[0][2]

# 	strs = [ [] for i in range(dense_shape[0]) ]

# 	string = []
# 	ptr = 0
# 	b = 0

# 	for i in range(len(indices)):
# 		if indices[i][0] != b:
# 			strs[b] = string
# 			string = []
# 			b = indices[i][0]

# 		string.append(values[ptr])

# 		ptr = ptr + 1

# 	strs[b] = string

# 	return strs






# def normalize(image):
#   	return (255. - image)/255.


# def resize(image, height):
# 	width = int(float(height * image.shape[1]) / image.shape[0])
# 	sample_img = cv2.resize(image, (width, height))
# 	return sample_img


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model_zip_path="/home/hilnels/mysite/Semantic-Model.zip"
# extract model files from zip
with ZipFile(model_zip_path, 'r') as zipObj:
   zipObj.extractall()



voc_path = "/home/hilnels/mysite/vocabulary_semantic.txt"
model_path = "/home/hilnels/mysite/Semantic-Model-2/semantic_model.meta"



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

@app.route('/img/<filename>')
def send_img(filename):
	return send_from_directory('', filename)

@app.route("/")
def root():
	return render_template('demo-index.html')

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
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


        print("made it to beginning of prediction")
        file_name = "/home/hilnels/mysite/.uploads/" + filename
        print("next line 1")
        image = cv2.imread(file_name,False)
        print("next line 2")
        image = ctc_utils.resize(image, HEIGHT)
        print("next line 3")
        image = ctc_utils.normalize(image)
        print("next line 4")
        image = np.asarray(image).reshape(1,image.shape[0],image.shape[1],1)
        print("next line 5")
        seq_lengths = [ image.shape[2] / WIDTH_REDUCTION ]
        print("next line 5")
        prediction = sess.run(decoded,
                              feed_dict={
                                  input: image,
                                  seq_len: seq_lengths,
                                  rnn_keep_prob: 1.0
                              })
        print("next line 6")
        print(prediction)
        str_predictions = ctc_utils.sparse_tensor_to_strs(prediction)
        for w in str_predictions[0]:
            print(int2word[w])


    return render_template('demo-result.html')


if __name__=="__main__":
	app.run()





"""
@app.route('/')
@app.route('/index')
def index():
	return '[testing]'
"""