from django.urls import path
from app.core import views
from django.views.generic import RedirectView

urlpatterns = [
    path('', RedirectView.as_view(url='/home/', permanent=False)),
    path('home/', views.home, name='home'),
    path('validate/', views.validate, name='validate'),
    path('prediction/<int:pk>/', views.prediction_detail, name='prediction-detail'),
    path('prediction/new/', views.new_prediction, name='new_prediction'),
    path('prediction/delete/<int:pk>/', views.delete_prediction, name='delete_prediction')
]