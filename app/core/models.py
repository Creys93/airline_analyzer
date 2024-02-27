from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

# Class used to store the information of each prediction made by a user.
class Prediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='predictions')
    file_name = models.CharField(max_length=100)
    mse = models.FloatField()
    mae = models.FloatField()
    r2 = models.FloatField()
    inertia_path = models.CharField(max_length=100)
    silhouette_path = models.CharField(max_length=100)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = 'prediction'


# Class used to store the prediction results of a given prediction.
class PredictionResult(models.Model):
    prediction = models.ForeignKey(Prediction, on_delete=models.CASCADE, related_name='results')
    index = models.IntegerField()
    num_passengers = models.IntegerField()
    sales_channel = models.CharField(max_length=100)
    trip_type = models.CharField(max_length=100)
    purchase_lead = models.IntegerField()
    length_of_stay = models.IntegerField()
    flight_hour = models.IntegerField()
    flight_day = models.CharField(max_length=3)
    route = models.CharField(max_length=100)
    booking_origin = models.CharField(max_length=100)
    wants_extra_baggage = models.BooleanField()
    wants_preferred_seat = models.BooleanField()
    wants_in_flight_meals = models.BooleanField()
    flight_duration = models.FloatField()
    booking_complete = models.BooleanField()
    prediction_value = models.FloatField()

    class Meta:
        db_table = 'prediction_result'


# Class used to store the SVM model results of a given prediction.
class SVMResult(models.Model):
    accuracy = models.FloatField()
    confusion_matrix_base64 = models.TextField()
    classification_report = models.TextField()
    cv_mean_accuracy = models.FloatField()

    # Relación con la predicción (puedes ajustar esto según tus modelos actuales)
    prediction = models.ForeignKey('Prediction', on_delete=models.CASCADE, related_name='svm_results')

    class Meta:
        db_table = 'prediction_svm_result'

