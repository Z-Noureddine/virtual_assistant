from django.shortcuts import render
from django.shortcuts import render
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.core import serializers
from django.conf import settings
import json
import os,numpy
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.backend import set_session
import librosa

max_length=29661
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

sess = tf.Session()
graph = tf.get_default_graph()

# IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras! 
# Otherwise, their weights will be unavailable in the threads after the session there has been set
set_session(sess)
model = load_model('/home/geek/Music/model_saved')

def predict_file(array):
	global graph
	global sess
	set_session(sess)
	x=np.array(array)
	x=(x-min(x))/(max(x)-min(x))
	file=x.astype(numpy.float16)
	file=np.pad(file,(0,max_length-len(file)),'constant').reshape(1,max_length)
	file=file.reshape(1,max_length,1)
	with graph.as_default():
		set_session(sess)
		predictions=model.predict(file)
	return np.argmax(predictions)


# Create your views here.
def hello(request):
    return render(request, 'hello.html', {})
'''

from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse,HttpResponse
from django.views.decorators.csrf import csrf_exempt
import os
from django.views import View
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from django.core.files.storage import default_storage
class GetFile(View):

	def get(self,request):   
	    return JsonResponse(status=200,data={},safe=False)

	# @csrf_exempt  
	def post(self,request):  
	    wavfile=request.FILES['wavfile']  
	    #logic of your wav file what you want to do 
	    return JsonResponse(status=200,data={'success':'success'},safe=False)
'''
@api_view(["POST"])
def responde_to_seq(seq):
	response=json.loads(seq.body)
	return JsonResponse("response:"+response,safe=False)
@api_view(["GET"])
def welcome(request):
	params = dict(request.query_params)

	#content = {"message": 'AAAA'+str(predict_file(np.array(params['seq'].split('_'))))+'AAA'}
	content = {"response": str(predict_file(np.array(list(map(lambda x: int(x), params['seq'][0].split('_'))))))}
	return JsonResponse(content)