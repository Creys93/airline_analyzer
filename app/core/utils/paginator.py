from django.core.paginator import Paginator
from django.db.models import Q
from datetime import datetime

def paginate(request, elements_per_page, defaultOrder, defaultOrderType, predictionResults=None):
    """
    Paginates a queryset based on the provided parameters.

    Args:
        request: HttpRequest object.
        elements_per_page: Number of elements per page.
        defaultOrder: Default order field.
        defaultOrderType: Default order type ('asc' or 'desc').
        predictionResults: Optional queryset to paginate.

    Returns:
        Paginated data.
    """
    order_by = request.GET.get('orderBy', defaultOrder)
    order_type = request.GET.get('orderType', defaultOrderType)
    page = int(request.GET.get('page', 1))
    search_query = request.GET.get('search_query')
    from_value = request.GET.get('from')
    extra_bag_value = request.GET.get('extra-bag')
    pref_seat_value = request.GET.get('pref-seat')
    meals_value = request.GET.get('meals')
    complete_value = request.GET.get('complete')

    data = {}

    if predictionResults is None:
        if not search_query:
            query = request.user.predictions.all()
        else:
            query = request.user.predictions.all().filter(
                Q(file_name__icontains=search_query) |
                Q(mse__icontains=search_query) |
                Q(mae__icontains=search_query) |
                Q(r2__icontains=search_query)
            )
        if from_value:
            from_date = datetime.strptime(from_value, '%Y-%m-%d')
            query = query.filter(created_at__gte=from_date)
    else:
        if not search_query:
            query = predictionResults.all()
        else:
            query = predictionResults.all().filter(
                Q(index__icontains=search_query) |
                Q(num_passengers__icontains=search_query) |
                Q(sales_channel__icontains=search_query) |
                Q(trip_type__icontains=search_query) |
                Q(purchase_lead__icontains=search_query) |
                Q(length_of_stay__icontains=search_query) |
                Q(flight_hour__icontains=search_query) |
                Q(flight_day__icontains=search_query) |
                Q(flight_duration__icontains=search_query) |
                Q(route__icontains=search_query) |
                Q(booking_origin__icontains=search_query) |
                Q(prediction_value__icontains=search_query)
            )

        if extra_bag_value:
            query = query.filter(wants_extra_baggage=True)
        if pref_seat_value:
            query = query.filter(wants_preferred_seat=True)
        if meals_value:
            query = query.filter(wants_in_flight_meals=True)
        if complete_value:
            query = query.filter(booking_complete=True)

    if order_type == 'asc':
        query = query.order_by(order_by)
    else:
        query = query.order_by('-' + order_by)

    data['paginator'] = {}
    data['paginator']['current_page'] = page
    data['paginator']['order_by'] = order_by
    data['paginator']['order_type'] = order_type

    # Create a Paginator object
    paginator = Paginator(query, elements_per_page)

    data['paginator']['num_pages'] = range(1, paginator.num_pages + 1)

    # Get the page number from the URL, or set it to 1 by default

    data['paginator']['results'] = paginator.get_page(page)
    data['paginator']['total_results'] = len(paginator.object_list)
    return data
