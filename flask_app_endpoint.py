from flask import Flask, request,send_from_directory,render_template
import os
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import boto3
import json
import io
import ctc_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

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

voc_file = "/home/hilnels/mysite/vocabulary_semantic.txt"

# Read the dictionary
dict_file = open(voc_file,'r')
dict_list = dict_file.read().splitlines()
int2word = dict()
for word in dict_list:
	word_idx = len(int2word)
	int2word[word_idx] = word
dict_file.close()

#AWS
client = boto3.client('sagemaker-runtime', aws_access_key_id="AKIARTTHSL7RJDWMHVE3", aws_secret_access_key="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")

@app.route('/img/<filename>')
def send_img(filename):
	return send_from_directory('', filename)

@app.route("/")
def root():
	return render_template('demo-index.html')

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
	if request.method == 'POST':
		print("hi")
		input_file = '/home/hilnels/mysite/.uploads/test1.png'

		width_reduction = 16
		seq_lengths = [54]

		image = cv2.imread(input_file, 2)
		image = ctc_utils.resize(image, 128)
		image = ctc_utils.normalize(image)
		image = np.asarray(image).reshape(1,image.shape[0],image.shape[1],1)

		seq_lengths = [ image.shape[2] / width_reduction ]



		body = json.dumps({"signature_name":'predict', "inputs": {"input": image.tolist(),
																"seq_len": [54],
																	"rnn_keep_prob": 1.0}})


		ioc_predictor_endpoint_name = 'tensorflow-inference-2021-04-01-07-57-51-933'
		content_type = 'application/json'
		ioc_response = client.invoke_endpoint(
			EndpointName=ioc_predictor_endpoint_name,
			Body=body,
			ContentType=content_type
		)

		response = ioc_response["Body"].read()
		data = json.loads(response.decode("utf-8"))
		list_tensor = data["outputs"]["fully_connected/BiasAdd"]
		output_tensor = tf.convert_to_tensor(list_tensor, tf.float32)
		decoded, _ = tf.nn.ctc_greedy_decoder(output_tensor, [54])
		sparse_tensor_value = [decoded[0].eval()]
		str_predictions = ctc_utils.sparse_tensor_to_strs(sparse_tensor_value)
		for w in str_predictions[0]:
			print(int2word[w])


		# array_of_notes = []

		# for w in str_predictions[0]:
		# 	array_of_notes.append(int2word[w])
		# print(array_of_notes)
		# test_output = array_of_notes[0]
		# notes=[]
		# for i in array_of_notes:
		# 	if i[0:5]=="note-":
		# 		if not i[6].isdigit():
		# 			notes.append(i[5:7])
		# 		else:
		# 			notes.append(i[5])

		# img = Image.open(img).convert('L')
		# size = (img.size[0], int(img.size[1]*1.5))
		# layer = Image.new('RGB', size, (255,255,255))
		# layer.paste(img, box=None)
		# img_arr = np.array(layer)
		# height = int(img_arr.shape[0])
		# width = int(img_arr.shape[1])
		# # print(img_arr.shape[0])
		# draw = ImageDraw.Draw(layer)
		# # font = ImageFont.truetype(<font-file>, <font-size>)
		# font = ImageFont.truetype("Aaargh.ttf", 20)
		# # draw.text((x, y),"Sample Text",(r,g,b))
		# j = width / 9
		# for i in notes:
		# 	draw.text((j, height-40), i, (0,0,0), font=font)
		# 	j+= (width / (len(notes) + 4))
		# layer.save("annotated.png")
		return render_template('demo-result.html', x = test_output)

if __name__=="__main__":
	app.run()





"""
@app.route('/')
@app.route('/index')
def index():
	return '[testing]'
"""
