from django.db import models
from django.core.exceptions import ValidationError
from django.conf import settings
from django.contrib.auth.models import AbstractUser

# class BrainScans(models.Model):
#     name        = models.CharField(max_length=200)
#     image       = models.ImageField(upload_to='images')
#     imageName   = models.CharField(max_length=200, editable=False, null=True, blank=True, unique=True)
#     prediction  = models.FloatField(editable=False, name="prediction", null=True, max_length=5, blank=True)
#     timestamp   = models.DateTimeField(auto_now_add=True)

#     # doctor = models.ForeignKey('NewDoctor', on_delete=models.CASCADE, related_name='brain_scans')

#     def clean(self):
#         """
#         Overriding clean method to perform custom validation
#         """
#         if BrainScans.objects.filter(imageName=self.imageName).exclude(pk=self.pk).exists():
#             raise ValidationError('An image with this filename already exists.')

#     def __str__(self):
#         return self.name
#     class Meta:
#         app_label  = 'cv'

class Contact(models.Model):
	Name        =models.CharField(max_length=200)
	Email       =models.EmailField()
	Message     =models.TextField()
	Subject     =models.TextField()
	"""docstring for Contact"""
	def __str__(self):
		return self.Name
     


class NewDoctor(models.Model):
    username = models.CharField(max_length=200, unique=True)
    fname = models.CharField(max_length=200)
    lname = models.CharField(max_length=200)
    Email = models.EmailField()
    Password = models.TextField()
    Password1 = models.TextField()

    def __str__(self):
        return self.username

class BrainScans(models.Model):

    name = models.CharField(max_length=200)
    image = models.ImageField(upload_to='images')
    imageName = models.CharField(max_length=200, null=True, blank=True)
    prediction = models.FloatField(null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    # Define the function inside the class
    def get_default_doctor_id():
        doctor = NewDoctor.objects.first()
        if doctor is not None:
            return doctor.id
        else:
            # Create a new doctor if none exists
            new_doctor = NewDoctor.objects.create(
                username='default',
                fname='Default',
                lname='Doctor',
                email='default@hospital.com',
                password='securepassword123'
            )
            return new_doctor.id

    doctor = models.ForeignKey(
        NewDoctor,
        on_delete=models.CASCADE,
        related_name='brain_scans',
        default=get_default_doctor_id  # Reference the function after it's defined
    )

    def clean(self):
        """
        Overriding clean method to perform custom validation
        """
        if BrainScans.objects.filter(imageName=self.imageName).exclude(pk=self.pk).exists():
            raise ValidationError('An image with this filename already exists.')

    def __str__(self):
        return self.name
    class Meta:
        app_label  = 'cv'
    
# Define the function before the BrainScans model
def get_default_doctor_id():
    doctor = NewDoctor.objects.first()
    if doctor is not None:
        return doctor.id
    else:
        # Create a new doctor if none exists
        new_doctor = NewDoctor.objects.create(
            username='default',
            fname='Default',
            lname='Doctor',
            email='default@hospital.com',
            password='securepassword123'
        )
        return new_doctor.id


# class User(AbstractUser):
#     is_doctor = models.BooleanField(default=False)
#     password_changed = models.BooleanField(default=False) 

#     groups = models.ManyToManyField(
#         'auth.Group',
#         verbose_name='groups',
#         blank=True,
#         help_text='The groups this user belongs to.',
#         related_name='custom_user_groups',  # Change here
#     )

#     user_permissions = models.ManyToManyField(
#         'auth.Permission',
#         verbose_name='user permissions',
#         blank=True,
#         help_text='Specific permissions for this user.',
#         related_name='custom_user_permissions',  # Change here
#     )