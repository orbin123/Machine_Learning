### Mapping Views  with url
from django.urls import path 
from . import views 

###local:8000/studyplanner/index
urlpatterns=[
    path('index', views.index),
    path('<day>', views.weekly_timetable)
]