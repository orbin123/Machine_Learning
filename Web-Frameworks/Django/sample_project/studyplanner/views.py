from django.shortcuts import render
from django.http import HttpResponse 

# Create your views here.
### Functions or classes are mapped to urls

def index(request):
    return HttpResponse('Welcome to my Timetable')

def monday(request):
    return HttpResponse('Today I am Learning Model Deployment')

def weekly_timetable(request, day):
    text=''
    if day=='monday':
        text='I will learn Model Deployment'
    elif day=='tuesday':
        text = 'I will learn exporting models'
    elif day=='wednesday':
        text = 'I will learn Flask and API Integration'
    elif day=='thursday':
        text = 'I will learn Django and djangorestframework'
    elif day=='friday':
        text = 'I will learn Docker'
    elif day=='saturday':
        text = 'I will learn Docker Compose'
    else:
        text = 'Sunday is a Holiday'
    return HttpResponse(text)
    