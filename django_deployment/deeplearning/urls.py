from django.urls import path#, include
from . import views
from django.conf import settings
from django.conf.urls.static import static

# Create your views here.
urlpatterns = [
    path('', views.home, name='site-home'),
    path('about-me/', views.about, name='about-me'),
    path('predict', views.predict, name='predict'),
    path('PlantVillage/', views.plant, name='PlantVillage'),

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
