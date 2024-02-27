from django.http import HttpResponseBadRequest, JsonResponse
import pandas as pd

valid_data_types = {
    'num_passengers': int,
    'sales_channel': str,
    'trip_type': str,
    'purchase_lead': int,
    'length_of_stay': int,
    'flight_hour': int,
    'flight_day': str,
    'route': str,
    'booking_origin': str,
    'wants_extra_baggage': int,
    'wants_preferred_seat': int,
    'wants_in_flight_meals': int,
    'flight_duration': float,
    'booking_complete': int
}

valid_types = {
    'int64': int,
    'object': str,
    'float64': float,
}

def validate_csv(file):

    if not is_valid_csv(file):
        return HttpResponseBadRequest("The selected file is not a valid CSV file.")
    
    if not are_columns_valid(file):
        return HttpResponseBadRequest("The selected file does not contain valid values.")
    
    return JsonResponse({'message': 'Valid CSV file'}, status=200)


def is_valid_csv(file):
    return file.name.endswith('.csv')

def are_columns_valid(file):
    df = pd.read_csv(file)

    required_columns = set(valid_data_types.keys())

    # Check required columns
    if not required_columns.issubset(df.columns):
        return False
    
    # Check data types of each column
    for index, row in df.iterrows():
        for column, expected_type in valid_data_types.items():
            value = row[column]
            if not isinstance(value, expected_type):
                return False

    return True
