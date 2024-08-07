from django.urls import path
from . import views 

urlpatterns = [
    # path('', views.index, name='index'),
    path('', views.landing, name="landing"),
    path('predict/', views.getPredictions, name='prediction'),
    path('predict-new/', views.getPredictionsNew, name='predictionNew'),
    path('predict-new-with-img-classifier/', views.getPredictionsNewWithImgClassifier, name='predictionNewWithImgClassifier'),
    path('predictionHome/', views.getPredictionsNewWithImgClassifier, name='predictionHome'),
    path('history/', views.BrainScansListView.as_view(), name='history'),
    path('login/', views.login, name="login"),
    path('dashboard/', views.dashboard, name="dashboard"),
    path('prognosis/', views.prognosis, name="prognosis"),
    path('segmentation_block/', views.segmentation_block, name="segmentation_block"),
    path('newdoctor/', views.newdoctor, name="newdoctor"),
    path('doctorlogin/', views.doctorlogin, name="doctorlogin"),
    path('doctordashboard/', views.doctordashboard, name="doctordashboard"),
    path('changepassword/', views.changepassword, name="changepassword"),
    path('dchangepassword/', views.dchangepassword, name="dchangepassword"),
    path('doctorlist/', views.doctorlist, name="doctorlist"), 
    path('delete-doctor/<str:username>/', views.delete_doctor, name='delete_doctor'),
    path('contacts/', views.contact_list, name='contact_list'),
]
