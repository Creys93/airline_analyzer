from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('', include('app.core.urls')),
    path('admin/', admin.site.urls),
    path('user/', include('app.user.urls')),
]
