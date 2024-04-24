from django.db import models
# Create your models here.
# Piezīmes modulis
class Post(models.Model):
    title= models.CharField(max_length=300)
    text= models.TextField()
    type= models.TextField()

