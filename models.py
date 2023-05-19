from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    first_name = None
    last_name = None
    name = models.CharField(max_length=100)

    def __str__(self) -> str:
        return self.username