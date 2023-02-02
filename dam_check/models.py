from django.db import models

# Create your models here.
class PixUpload(models.Model):
    imagefile = models.ImageField(upload_to='pix_upload', blank=True)
    # this help save the datase even is no image is passed