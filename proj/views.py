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



import os,random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras import Sequential,Input
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense,Flatten
import pickle
import numpy as np
import pandas as pd



greetings=pd.read_csv('/home/geek/Music/Virtual_assisstant_MD/conversation/greetings.csv')

word2int_type=pickle.load(open('/home/geek/Music/Virtual_assisstant_MD/word2int_type.pkl','rb'))
word2int_conv=pickle.load(open('/home/geek/Music/Virtual_assisstant_MD/conversation/word2int_conv.pkl','rb'))


max_length_type=10
max_length_conv=8

sess = tf.Session()
graph = tf.get_default_graph()

# IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras! 
# Otherwise, their weights will be unavailable in the threads after the session there has been set
set_session(sess)
model = models.Sequential([
    layers.Input(shape=(max_length,1)),
    layers.Conv1D(64, 2, activation='tanh'),
    layers.Conv1D(32, 2, activation='tanh'),
    layers.MaxPooling1D(pool_size=4,strides=4),
    layers.Conv1D(32, 2, activation='tanh'),
    layers.Conv1D(32, 2, activation='tanh'),
    layers.MaxPooling1D(pool_size=4,strides=4),
    layers.Dropout(0.25),
    layers.Conv1D(32, 2, activation='tanh'),
    layers.Conv1D(32, 2, activation='tanh'),
    layers.MaxPooling1D(pool_size=3,strides=3),
    layers.Conv1D(32, 2, activation='tanh'),
    layers.Conv1D(32, 2, activation='tanh'),
    layers.MaxPooling1D(pool_size=2,strides=2),
    layers.Dropout(0.1),
    layers.Flatten(),
    layers.Dense(16, activation='tanh'),
    layers.Dense(8, activation='tanh'),
    layers.Dense(4, activation='softmax'),
])
model.load_weights(str(Path(__file__).resolve().parent)+'/speech_recog.h5')

#model 1
type_detect = Sequential()
type_detect.add(Embedding(len(word2int_type.keys())+2, 8, input_length=max_length_type,trainable=True))
type_detect.add(Flatten())
type_detect.add(Dense(16, activation='tanh'))
type_detect.add(Dense(8, activation='tanh'))
type_detect.add(Dense(4, activation='softmax'))
type_detect.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

type_detect.load_weights('/home/geek/Music/Virtual_assisstant_MD/type demande/type_demande.h5')

#model 2
conv_model = Sequential()
conv_model.add(Embedding(len(word2int_conv.keys())+2, 8, input_length=max_length_conv,trainable=True))
conv_model.add(Flatten())
conv_model.add(Dense(6, activation='tanh'))
conv_model.add(Dense(4, activation='tanh'))
conv_model.add(Dense(2, activation='softmax'))
conv_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

conv_model.load_weights('/home/geek/Music/Virtual_assisstant_MD/conversation/conversation.h5')

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
def chat(request):
    return render(request, 'chat.html', {})
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
	filename=os.getcwd()+"/"+"audio_{}".format(random.randint(0,5000))
	print(filename)
	f=open(filename+'.webm','wb')
	f.write(request.FILES['audio'].read())
	f.close()
	os.system('ffmpeg -i "{}.webm" -vn "{}.wav"'.format(filename,filename))

	prediction=predict_file(filename+'.wav')
	content = {"response": str(prediction)}
	#os.system('rm {}*'.format(filename))
	return JsonResponse(content)







############################################################
############################################################
############################################################




results={0:"ﻋﻘﺪ اﺯﺩﻳﺎﺩ", 
1:"ﺗﺼﺤﻴﺢ اﻻﻣﻀﺎء"  ,           
2:"ﺗﺼﺮﻳﺢ ﺑﺎﻟﺸﺮﻑ",           
3           :"Autre"}
def predict_type(seq):
    tok=[]
    for word in seq.split():
        tok+=[word2int_type[word]] if word in word2int_type.keys() else [118]
    tok+= [0]*(max_length_type-len(tok))
    tok = np.array(tok).reshape(1,10)
    with graph.as_default():
    	set_session(sess)
    	typ=type_detect.predict(tok)
    return np.argmax(typ)



def predict_conv(seq):
    tok=[]
    for word in seq.split():
        tok+=[word2int_conv[word]] if word in word2int_conv.keys() else [32]
    tok+= [0]*(max_length_conv-len(tok))
    tok=np.array(tok).reshape(1,max_length_conv)
    with graph.as_default():
    	set_session(sess)
    	prd=conv_model.predict(tok)
    type_seq=np.argmax(prd)+1
    results=greetings[greetings['type']==type_seq]
    return results['reponse'].sample().to_string(index=False)
def submit_data(data,request_type):
  pass
  estimated_time=random.randint(5,20)
  return estimated_time
@api_view(["GET"])
def chatting(request):
	params = dict(request.query_params)
	seq=params['seq'][0].replace('*',' ')
	_type=predict_type(seq)
	if _type == 3:
		resp=predict_conv(seq)
	else:
	    resp="type"+results[_type]
	    required_data={"اﻻﺳﻢ":"","اﻟﻨﺴﺐ":"","ﺭﻗﻢ اﻟﺒﺎﻃﺎﻗﺔ اﻟﻮﻃﻨﻴﺔ":"","اﻟﺒﺮﻳﺪ اﻻﻟﻜﺘﺮﻭﻧﻲ":""}
	    es_time=submit_data(required_data,_type)
	    #resp="{} ﺩﻳﺎﻟﻜﻢ ﻏﺎﻳﻜﻮﻥ ﻭاﺟﺪ ﺧﻼﻝ {} ﺩﻗﻴﻘﺔ".format(results[_type],es_time)
	prediction=seq
	content = {"response": str(resp)}
	#os.system('rm {}*'.format(fiename))
	return JsonResponse(content)





