from django.urls import path
from proj import views
from django.conf.urls import url

urlpatterns = [
    path('', views.hello, name='hello'),
    path('chat', views.chat, name='chat'),
    path('test', views.test, name='test'),
    url('responde_to_seq/',views.responde_to_seq),
    path('api/', views.welcome),
    path('chatting/', views.chatting),
]

