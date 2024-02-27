from django.contrib.auth.decorators import login_required
from django.http import HttpResponseBadRequest
from django.shortcuts import render, redirect, get_object_or_404
from app.core.utils.constants import PREDICTION_COLUMNS, PREDICTION_RESULTS_COLUMNS
from app.core.utils.validator import validate_csv
from app.core.utils.processor import preprocessing
from app.core.utils.paginator import paginate
from .models import Prediction

@login_required()
def home(request):
    """
    Renders the home page with a list of predictions.

    Args:
        request: HttpRequest object.

    Returns:
        Rendered HTML template with prediction data.
    """
    data = paginate(request, 25, 'created_at', 'desc')
    return render(request, 'prediction-list.html', {'data': data, 'columns': PREDICTION_COLUMNS})

@login_required()
def validate(request):
    """
    Validates a CSV file uploaded through a POST request.

    Args:
        request: HttpRequest object.

    Returns:
        HttpResponseBadRequest if the request method is not POST or no file is uploaded,
        otherwise the result of CSV validation.
    """
    if request.method == 'POST' and request.FILES.get('filename'):
        return validate_csv(request.FILES.get('filename'))
    return HttpResponseBadRequest("Expected a POST request with a CSV file.")

@login_required()
def new_prediction(request):
    """
    Handles the creation of a new prediction based on an uploaded CSV file.

    Args:
        request: HttpRequest object.

    Returns:
        Redirects to the home page if the prediction is successfully processed,
        otherwise renders the new prediction form.
    """
    if request.method == 'POST' and request.FILES.get('filename'):
        preprocessing(request.FILES.get('filename'), request.user)
        return redirect('home')
    else:
        return render(request, 'new-prediction.html')

@login_required()
def prediction_detail(request, pk):
    """
    Renders the detail page for a specific prediction.

    Args:
        request: HttpRequest object.
        pk: Primary key of the prediction to display.

    Returns:
        Rendered HTML template with prediction details.
    """
    prediction = get_object_or_404(Prediction, pk=pk)
    data = paginate(request, 2500, 'index', 'asc', prediction.results)
    svm_result = prediction.svm_results.get()
    classification_lines = svm_result.classification_report.split('\n')[2:]
    classification_report = [line.split() for line in classification_lines]
    svm_data = {
        'svm_result': svm_result,
        'svm_classification_report': classification_report,
    }
    return render(request, 'prediction_detail.html', {'prediction': prediction, 'data': data, 'columns': PREDICTION_RESULTS_COLUMNS, 'svm_data': svm_data })

@login_required()
def delete_prediction(request, pk):
    """
    Deletes a prediction.

    Args:
        request: HttpRequest object.
        pk: Primary key of the prediction to delete.

    Returns:
        Redirects to the home page after deleting the prediction.
    """
    prediction = get_object_or_404(Prediction, pk=pk)
    if request.method == 'POST':
        prediction.delete()
    return redirect('home')
