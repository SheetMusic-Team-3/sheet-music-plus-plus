from flask import Flask, request,send_from_directory,render_template
import os
import tensorflow as tf
from tensorflow.python.framework import ops
import tensorflow.compat.v1 as tfc
import cv2
import numpy as np
from zipfile import ZipFile
from PIL import Image, ImageFont, ImageDraw
import glob
import ctc_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

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






def normalize(image):
  	return (255. - image)/255.


def resize(image, height):
	width = int(float(height * image.shape[1]) / image.shape[0])
	sample_img = cv2.resize(image, (width, height))
	return sample_img

model_zip_path="/home/hilnels/mysite/Semantic-Model.zip"
# extract model files from zip
with ZipFile(model_zip_path, 'r') as zipObj:
    zipObj.extractall()

voc_path = "/home/hilnels/mysite/vocabulary_semantic.txt"
model_path = "/home/hilnels/mysite/Semantic-Model-2/semantic_model.meta"

# ops.reset_default_graph()
# sess = tfc.InteractiveSession()
# # Read the dictionary
# dict_file = open(voc_file,'r')
# dict_list = dict_file.read().splitlines()
# int2word = dict()
# for word in dict_list:
# 	word_idx = len(int2word)
# 	int2word[word_idx] = word
# dict_file.close()

# # Restore weights
# tfc.disable_v2_behavior()
# saver = tfc.train.import_meta_graph(model)
# saver.restore(sess,model[:-5])
# # saver.restore(sess, tf.train.latest_checkpoint('./'))

# graph = ops.get_default_graph()


# input = graph.get_tensor_by_name("model_input:0")
# seq_len = graph.get_tensor_by_name("seq_lengths:0")
# rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
# height_tensor = graph.get_tensor_by_name("input_height:0")
# width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
# logits = tfc.get_collection("logits")[0]


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
print(decoded)

@app.route('/img/<filename>')
def send_img(filename):
	return send_from_directory('', filename)

@app.route("/")
def root():
	return render_template('demo-index.html')

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
	if request.method == 'POST':
		f = request.files['file']
		img = f
		image = Image.open(img).convert('L')
		image = np.array(image)
		image = ctc_utils.resize(image, HEIGHT)
		image = ctc_utils.normalize(image)
		image = np.asarray(image).reshape(1,image.shape[0],image.shape[1],1)

		print("starting predict function after image setup")

		seq_lengths = [ image.shape[2] / WIDTH_REDUCTION ]

		prediction = sess.run(decoded, feed_dict = {
			input: image,
			seq_len: seq_lengths,
			rnn_keep_prob: 1.0,
		})

		print(prediction)

# 		str_predictions = sparse_tensor_to_strs(prediction)

		str_predictions = ctc_utils.sparse_tensor_to_strs(prediction)
		for w in str_predictions[0]:
			print(int2word[w])

# 		print(f"str_predictions {str_predictions}")

# 		array_of_notes = []

# 		for w in str_predictions[0]:
# 			array_of_notes.append(int2word[w])
# 		notes=[]
# 		for i in array_of_notes:
# 			if i[0:5]=="note-":
# 				if not i[6].isdigit():
# 					notes.append(i[5:7])
# 				else:
# 					notes.append(i[5])

# 		print(f"array_of_notes {array_of_notes}")

# 		img = Image.open(img).convert('L')
# 		size = (img.size[0], int(img.size[1]*1.5))
# 		layer = Image.new('RGB', size, (255,255,255))
# 		layer.paste(img, box=None)
# 		img_arr = np.array(layer)
# 		height = int(img_arr.shape[0])
# 		width = int(img_arr.shape[1])
# 		# print(img_arr.shape[0])
# 		draw = ImageDraw.Draw(layer)
# 		# font = ImageFont.truetype(<font-file>, <font-size>)
# 		font = ImageFont.truetype("Aaargh.ttf", 20)
# 		# draw.text((x, y),"Sample Text",(r,g,b))
# 		j = width / 9
# 		for i in notes:
# 			draw.text((j, height-40), i, (0,0,0), font=font)
# 			j+= (width / (len(notes) + 4))
# 		layer.save("annotated.png")
# 		return render_template('demo-result.html')



if __name__=="__main__":
	app.run()





"""
@app.route('/')
@app.route('/index')
def index():
	return '[testing]'
"""