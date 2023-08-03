from django import forms

from .models import UploadedImage

class PhotoForm(forms.ModelForm):
    class Meta:
        model = UploadedImage
        fields = ('image', )
