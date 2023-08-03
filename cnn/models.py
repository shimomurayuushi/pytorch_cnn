from django.db import models


class UploadedImage(models.Model):
    image = models.ImageField(upload_to='uploads/')

    def __str__(self):
        return f"Uploaded Image: {self.image}"