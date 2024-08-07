import random
from django.forms import ValidationError
from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.views.generic.list import ListView
from django.db import IntegrityError
from django.shortcuts import redirect, render
from django.contrib.auth.models import User  

from .models import Contact, NewDoctor
from django.contrib.auth.models import auth,User
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth import login
from .model import ModelDeployment
from .forms import *
from .models import BrainScans
from . import imageClassifier

import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image
from django.contrib import messages
from django.urls import reverse_lazy


def doctorlist(request):
    if not request.user.is_superuser:  # Check if the user is an admin/superuser
        return redirect('doctordashboard')  # Redirect if not

    doctors = NewDoctor.objects.all()  
    return render(request, 'doctorlist.html', {'doctors': doctors})



# def index(request):
#     return HttpResponse("<h1>Hello, world. You're at the cv index.</h1><h3>Try to use paths:<br>- /predict</h3>")

def getPredictions(request):
    if request.method == 'POST':
        form = BrainScansForm(request.POST, request.FILES)

        if form.is_valid():
            # return HttpResponse(ModelDeployment.get_prediction_from_image("cv/model/images/image.jpeg"))
            # return HttpResponse(ModelDeployment.get_prediction_from_image_upload(request.FILES["brain_scan_img"]))
            print("request.FILES['brain_scan_img']")
            print(request.FILES['brain_scan_img'])
            print(request.FILES)
            prediction, imageName, original_image = ModelDeployment.get_prediction_from_image_upload(request.FILES["brain_scan_img"])

            form = BrainScansForm()
            return render(request, 'brain_scan_predict.html',
            {'form' : form,
            "prediction": round(prediction*100,2),
            "imageName": imageName,
            'original_image': original_image.decode('utf-8')}
            )
    else:
        form = BrainScansForm()
        return render(request, 'brain_scan_predict.html', {'form' : form})

def getPredictionsNew(request):
    last_prediction = BrainScans.objects.last()
    if last_prediction and last_prediction.imageName:
        last_prediction.imageName = last_prediction.imageName[:20]

    if request.method == 'POST':
        form = BrainScansFormNew(request.POST, request.FILES)

        if form.is_valid():
            try:
                form.save() 
                filename = form.instance.image.file.name
                prediction, imageName, original_image = ModelDeployment.get_prediction_from_image_upload_new(filename)
                form.instance.prediction = round(prediction*100,2)
                form.instance.imageName = imageName
                form.save()

            except ValidationError as e:
                form.add_error('image', e)  

            # Render the template regardless of success or error
            return render(request, 'brain_scan_predict-new.html',
                          {'form': form,
                           "prediction": round(prediction*100,2) if 'prediction' in locals() else None,  # Display prediction only if it exists
                           "imageName": imageName if 'imageName' in locals() else None,
                           'original_image': original_image.decode('utf-8') if 'original_image' in locals() else None,
                           "last_prediction": last_prediction})

    else:
        form = BrainScansFormNew()
        return render(request, 'brain_scan_predict-new.html', {'form': form, "last_prediction": last_prediction})


# def getPredictionsNew(request):
#     last_prediction = BrainScans.objects.last()
#     if last_prediction and last_prediction.imageName:
#         last_prediction.imageName = last_prediction.imageName[:20]

#     if request.method == 'POST':
#         form = BrainScansFormNew(request.POST, request.FILES)

#         if form.is_valid():
#             form.save()
#             filename = form.instance.image.file.name
#             prediction, imageName, original_image = ModelDeployment.get_prediction_from_image_upload_new(form.instance.image.file.name)
#             form.instance.prediction = round(prediction*100,2)
#             form.instance.imageName = str(request.FILES['image']) #filename
#             form.save()

#             form = BrainScansFormNew()
#             return render(request, 'brain_scan_predict-new.html',
#             {'form' : form,
#             "prediction": round(prediction*100,2),
#             "imageName": str(request.FILES['image']), # str(filename),
#             'original_image': original_image.decode('utf-8'),
#             "last_prediction": last_prediction}
#             )
#     else:
#         form = BrainScansFormNew()
#         return render(request, 'brain_scan_predict-new.html',
#         {'form' : form,
#             "last_prediction": last_prediction})

class BrainScansListView(ListView):

    model = BrainScans
    paginate_by = 100  # if pagination is desired
    def get_queryset(self):
        # Assuming 'doctor' is the logged-in doctor's username stored in the session
        doctor_username = self.request.session.get('doctor_username')
        if doctor_username:
            doctor = NewDoctor.objects.get(username=doctor_username)
            return BrainScans.objects.filter(doctor=doctor)
        return BrainScans.objects.none()  # Return an empty queryset if no doctor is logged in

    # def get_context_data(self, **kwargs):
    #     context = super().get_context_data(**kwargs)
    #     context['predict_page'] = timezone.now()
    #     return context

# def getPredictionsNewWithImgClassifier(request):
#     last_prediction = BrainScans.objects.last()
#     if last_prediction and last_prediction.imageName:
#         last_prediction.imageName = last_prediction.imageName[:20]

#     if request.method == 'POST':
#         form = BrainScansFormNew(request.POST, request.FILES)

#         if form.is_valid():
#             form.save()
#             # filename = form.instance.image.file.name

#             isMri = imageClassifier.get_class_of_image(form.instance.image.file.name)
#             print("isMri!!!!!!!!!!!!!!!!!!!")
#             print(isMri)
#             prediction, imageName, original_image = ModelDeployment.get_prediction_from_image_upload_new(form.instance.image.file.name)
#             form.instance.prediction = round(prediction*100,2)
#             form.instance.imageName = str(request.FILES['image']) #filename
#             if isMri:
#                 form.save()
#             else:
#                 prediction = False
#                 form.instance.delete()

#             form = BrainScansFormNew()
#             return render(request, 'brain_scan_predict-new-with-MRI-classifier.html',
#             {'form' : form,
#             "prediction": round(prediction*100,2),
#             "imageName": str(request.FILES['image']), # str(filename),
#             'original_image': original_image.decode('utf-8'),
#             'isNotMri': not isMri,
#             "last_prediction": last_prediction}
#             )
#     else:
#         form = BrainScansFormNew()
#         return render(request, 'brain_scan_predict-new-with-MRI-classifier.html',
#         {'form' : form,
#             "last_prediction": last_prediction})
    
def getPredictionsNewWithImgClassifier(request):
    last_prediction = BrainScans.objects.last()
    if last_prediction and last_prediction.imageName:
        last_prediction.imageName = last_prediction.imageName[:20]

    if request.method == 'POST':
        form = BrainScansFormNew(request.POST, request.FILES)

        if form.is_valid():
            try:
                form.save()
                isMri = imageClassifier.get_class_of_image(form.instance.image.file.name)

                if isMri:
                    filename = form.instance.image.file.name
                    prediction, imageName, original_image = ModelDeployment.get_prediction_from_image_upload_new(filename)
                    form.instance.prediction = round(prediction*100,2)
                    form.instance.imageName = imageName
                    form.save()
                    isNotumor = 'Te-no_' in filename  # Adjust according to how you determine the filename
                else:
                    prediction = False  # Not an MRI, set prediction to False for display
                    form.instance.delete()

            except ValidationError as e:
                form.add_error('image', e)  

            return render(request, 'brain_scan_predict-new-with-MRI-classifier.html',
                          {'form': form,
                           'prediction': prediction, 
                           'imageName': imageName if 'imageName' in locals() else None,
                           'original_image': original_image.decode('utf-8') if 'original_image' in locals() else None,
                           'isNotMri': not isMri if 'isMri' in locals() else False,
                           "last_prediction": last_prediction,
                           'isNotumor': isNotumor})

    else:
        form = BrainScansFormNew()
        return render(request, 'brain_scan_predict-new-with-MRI-classifier.html', {'form': form, "last_prediction": last_prediction})             




def login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        if username == 'admin' and password == 'admin':
            # Simulate successful authentication of 'admin' user
            request.session['is_admin'] = True  # Mark admin session
            return redirect('dashboard') 

        else:
            messages.error(request, 'Incorrect username or password')

    return render(request, 'cv/login.html')


def dashboard(request):
	return render(request, 'cv/dashboard.html')

def doctordashboard(request):
	return render(request, 'cv/doctordashboard.html')


import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image

def prognosis(request):
    data = [random.randint(4, 10) for _ in range(10)]  # Generate random data for both GET and POST
    last_prediction = BrainScans.objects.last()
    if last_prediction and last_prediction.imageName:
        last_prediction.imageName = last_prediction.imageName[:20]

    if request.method == 'POST':
        form = BrainScansFormNew(request.POST, request.FILES)
        if form.is_valid():
            try:
                form.save()
                isMri = imageClassifier.get_class_of_image(form.instance.image.file.name)
                if isMri:
                    filename = form.instance.image.file.name
                    prediction, imageName, original_image = ModelDeployment.get_prediction_from_image_upload_new(filename)
                    segmentation_mask = get_segmentation_mask(filename)
                    form.instance.prediction = round(prediction*100,2)
                    form.instance.imageName = imageName
                    form.save()
                else:
                    prediction = False
                    form.instance.delete()
            except ValidationError as e:
                form.add_error('image', e)

            return render(request, 'cv/prognosis.html', {
                'form': form,
                'prediction': prediction,
                'imageName': imageName if 'imageName' in locals() else None,
                'original_image': original_image.decode('utf-8') if 'original_image' in locals() else None,
                'isNotMri': not isMri if 'isMri' in locals() else False,
                "last_prediction": last_prediction,
                'data': data
            })

    else:
        form = BrainScansFormNew()
        return render(request, 'cv/prognosis.html', {'form': form, "last_prediction": last_prediction, 'data': data})          


def segmentation_block(request):
    last_prediction = BrainScans.objects.last()
    if last_prediction and last_prediction.imageName:
        last_prediction.imageName = last_prediction.imageName[:20]

    if request.method == 'POST':
        form = BrainScansFormNew(request.POST, request.FILES)

        if form.is_valid():
            try:
                form.save()
                isMri = imageClassifier.get_class_of_image(form.instance.image.file.name)

                # Inside your try block, after confirming the image is an MRI
                if isMri:
                    filename = form.instance.image.file.name
                    prediction, imageName, original_image = ModelDeployment.get_prediction_from_image_upload_new(filename)
                    segmentation_mask = get_segmentation_mask(filename)  # Call your segmentation function
                    form.instance.prediction = round(prediction*100,2)
                    form.instance.imageName = imageName
                    form.save()
                else:
                    prediction = False  # Not an MRI, set prediction to False for display
                    form.instance.delete()

            except ValidationError as e:
                form.add_error('image', e)  

            return render(request, 'cv/segmentation_block.html',
                          {'form': form,
                           'prediction': prediction, 
                           'imageName': imageName if 'imageName' in locals() else None,
                           'original_image': original_image.decode('utf-8') if 'original_image' in locals() else None,
                           'isNotMri': not isMri if 'isMri' in locals() else False,
                           "last_prediction": last_prediction})

    else:
        form = BrainScansFormNew()
        return render(request, 'cv/segmentation_block.html', {'form': form, "last_prediction": last_prediction})             



	# return render(request, 'cv/segmentation_block.html',  'imageName': imageName if 'imageName' in locals() else None)


# def newdoctor(request):
#     if request.method == 'POST':
#         username = request.POST.get('username')
#         fname = request.POST.get('fname')
#         lname = request.POST.get('lname') 
#         email = request.POST.get('email')
#         password = request.POST.get('password')
#         confirmation_password = request.POST.get('password1')

#         if password != confirmation_password:
#             messages.info(request, 'Passwords do not match.')
#             return render(request, 'cv/newdoctor.html')

#         if User.objects.filter(email=email).exists():
#             messages.info(request, 'Email already exists.')
#             return render(request, 'cv/newdoctor.html')

#         if User.objects.filter(username=username).exists():
#             messages.info(request, 'Username already exists.')
#             return render(request, 'cv/newdoctor.html')

#         # Create the user
#         user = User.objects.create_user(username=username, email=email, password=password)
#         user.save()

#         messages.success(request, 'Account created successfully!')  # Success message
#         return redirect('doctorlogin')

#     else:
#         return render(request, 'cv/newdoctor.html')

# Import the built-in User model

def landing(request):
    if request.method == "POST":
        contact     = Contact()
        Name        = request.POST.get('Name')
        Email       = request.POST.get('Email')
        Subject     = request.POST.get('Subject')
        Message     = request.POST.get('Message')
        contact.Name    = Name
        contact.Email   = Email
        contact.Subject = Subject
        contact.Message = Message
        contact.save()
    return render(request, "cv/landing.html")

def contact_list(request):
    contacts = Contact.objects.all()  # Fetch all contact records
    return render(request, 'cv/contacts.html', {'contacts': contacts})     


def newdoctor(request):
    if request.method == 'POST':
        newdoctor = NewDoctor()
        username = request.POST.get('username')
        fname = request.POST.get('fname')
        lname = request.POST.get('lname')
        Email = request.POST.get('email')
        Password = request.POST.get('password')
        Password1 = request.POST.get('password1')

        newdoctor.username = username
        newdoctor.fname = fname
        newdoctor.lname = lname
        newdoctor.Email = Email
        newdoctor.Password = Password
        newdoctor.Password1 = Password1

        if Password != Password1:
            messages.error(request, 'Passwords do not match.')
            return render(request, 'cv/newdoctor.html', {'password_mismatch': True})

        try:
            newdoctor.save()
            messages.success(request, 'Account created successfully!')
        except IntegrityError:
            messages.error(request, 'This username already exists.')
            return render(request, 'cv/newdoctor.html', {'username_exists': True})

    return render(request, 'cv/newdoctor.html', {'password_mismatch': False, 'username_exists': False})



# def doctorlogin(request):  # Renamed for clarity 
#     if request.method == 'POST':
#         username = request.POST['username']
#         password = request.POST['password']

#         try:
#             user = NewDoctor.objects.get(username=username) 

#             # Assuming plain text passwords (PLEASE DO NOT DO THIS IN PRODUCTION!)
#             if user.Password == password: 
#                 login(request) 
#                 return redirect('doctordashboard')  

#             else:
#                 return render(request, 'cv/doctorlogin.html', {'error_message': 'Invalid password'})

#         except NewDoctor.DoesNotExist:
#             return render(request, 'cv/doctorlogin.html', {'error_message': 'Invalid username'}) 

#     else:
#         return render(request, 'cv/doctorlogin.html')
def doctorlogin(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        try:
            doctor = NewDoctor.objects.get(username=username)
            if doctor.Password == password:
                request.session['doctor_username'] = username  # Store the doctor's username in the session
                return redirect('doctordashboard')
            else:
                return render(request, 'cv/doctorlogin.html', {'error_message': 'Invalid password'})
        except NewDoctor.DoesNotExist:
            return render(request, 'cv/doctorlogin.html', {'error_message': 'Invalid username'})
    else:
        return render(request, 'cv/doctorlogin.html')

def changepassword(request):
    username = request.GET.get('username', '')
    if request.method == 'POST':
        username = request.POST.get('username')
        new_password = request.POST.get('password')
        confirm_password = request.POST.get('password1')
        
        if new_password == confirm_password:
            try:
                doctor = NewDoctor.objects.get(username=username)
                # Directly assign the new password without hashing
                doctor.Password = new_password
                doctor.save()
                messages.success(request, 'Password updated successfully.')
                return redirect('changepassword')  
            except NewDoctor.DoesNotExist:
                messages.error(request, 'Doctor not found.')
        else:
            messages.error(request, 'Passwords do not match.')
    
    return render(request, 'cv/changepassword.html', {'username': username})


def dchangepassword(request):
    username = request.GET.get('username', '')
    if request.method == 'POST':
        username = request.POST.get('username')
        new_password = request.POST.get('password')
        confirm_password = request.POST.get('password1')
        
        if new_password == confirm_password:
            try:
                doctor = NewDoctor.objects.get(username=username)
                # Directly assign the new password without hashing
                doctor.Password = new_password
                doctor.save()
                messages.success(request, 'Password updated successfully.')
                return redirect('dchangepassword')  
            except NewDoctor.DoesNotExist:
                messages.error(request, 'Doctor not found.')
        else:
            messages.error(request, 'Passwords do not match.')
    
    return render(request, 'cv/dchangepassword.html', {'username': username})

def doctorlist(request):
    doctors = NewDoctor.objects.all()  # Fetch all doctor records
    return render(request, 'cv/doctorlist.html', {'doctors': doctors})



def delete_doctor(request, username):
    doctor = NewDoctor.objects.get(username=username)
    doctor.delete()
    messages.success(request, 'Doctor deleted successfully!')
    return redirect('doctorlist')

def doctor_dashboard(request):
    if request.user.is_authenticated:
        # Assuming the username of the doctor is stored in the user model
        doctor = NewDoctor.objects.get(username=request.user.username)
        return render(request, 'doctordashboard.html', {'doctor': doctor})
    else:
        # Redirect to login page or handle as per your flow
        return redirect('/doctorlogin/')
    



