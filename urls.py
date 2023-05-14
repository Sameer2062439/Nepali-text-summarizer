from django.urls import path
from .views import SummarizeView, IndexView, login_view, signup_view

urlpatterns = [
    path('', IndexView.as_view(), name='index'),
    path('summarize/', SummarizeView.as_view(), name='summarize'),
    path('login/', login_view, name='login'),
    path('signup/', signup_view, name='signup'),
]