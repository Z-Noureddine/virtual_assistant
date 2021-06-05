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
import pathlib,random
import numpy as np
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.backend import set_session
import librosa
from pathlib import Path

max_length=29661
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

sess = tf.Session()
graph = tf.get_default_graph()

# IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras! 
# Otherwise, their weights will be unavailable in the threads after the session there has been set
set_session(sess)
model = load_model(str(Path(__file__).resolve())+'/model_saved')

def predict_file(file):
	global graph
	global sess
	x,_ = librosa.load(file, sr=3000)
	x=(x-min(x))/(max(x)-min(x))
	file= x.astype(numpy.float16)
	file=np.pad(file,(0,max_length-len(file)),'constant').reshape(1,max_length)
	file=file.reshape(1,max_length,1)
	with graph.as_default():
		set_session(sess)
		predictions=model.predict(file)
	return np.argmax(predictions)


# Create your views here.
def hello(request):
    return render(request, 'hello.html', {})
def test(request):
    return render(request, 'index.html', {})
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
@api_view(["POST"])
def welcome(request):
	params = dict(request.query_params)
	filename="audio_{}".format(random.randint(0,5000))
	f=open(filename+'.webm','wb')
	f.write(request.FILES['audio'].read())
	f.close()
	os.system('ffmpeg -i "{}.webm" -vn "{}.wav"'.format(filename,filename))

	prediction=predict_file(filename+'.wav')
	content = {"response": str(prediction)}
	#os.system('rm {}*'.format(filename))
	return JsonResponse(content)