from django.urls import path
from . import views

urlpatterns = [
    path('', views.chatbot_view, name='chatbot_view'),
    path('query/', views.query_view, name='query_view'),  # Optional, can be removed if not used
]