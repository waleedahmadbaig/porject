from django.db import models  
from django.forms import fields  
from .models import BrainScans
from django import forms  


class BrainScansForm(forms.Form):
    brain_scan_img = forms.ImageField(label='MRI Scan')

class BrainScansFormNew(forms.ModelForm):
    class Meta:  
        # To specify the model to be used to create form  
        model = BrainScans  
        # It includes all the fields of model  
        fields = '__all__'
        labels = {
            'name': 'Your name',
            'image': 'Brain MRI scan'
        }

