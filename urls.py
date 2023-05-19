from django.urls import path
from .views import SummarizeView, IndexView, login_view, signup_view, LogoutView
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', IndexView.as_view(), name='index'),
    path('summarize/', SummarizeView.as_view(), name='summarize'),
    path('login/', login_view, name='login'),
    path('signup/', signup_view, name='signup'),
    path('logout/', LogoutView.as_view(), name='logout'),

]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)