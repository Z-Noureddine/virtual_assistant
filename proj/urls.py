from django.urls import path
from proj import views
from django.conf.urls import url

urlpatterns = [
    path('', views.hello, name='hello'),
    url('responde_to_seq/',views.responde_to_seq),
    path('api/', views.welcome),
]

