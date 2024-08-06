from django.http import HttpResponse
from django.shortcuts import render

def index(request):
    return HttpResponse("<h1>Welcome to the django computer vision deployment!</h1><h3>Try using paths:<br>- /admin<br>- /cv</h3>")
