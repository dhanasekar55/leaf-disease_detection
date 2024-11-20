from django.db import models

class PlantDisease(models.Model):
    image = models.ImageField(upload_to="diseases/")
    predicted_disease = models.CharField(max_length=100, null=True, blank=True)

    def __str__(self):
        return f"Disease: {self.predicted_disease}, Image: {self.image.url}"


class PageView(models.Model):
    page_name = models.CharField(max_length=255)
    view_count = models.IntegerField(default=0)

    def increment_view(self):
        """Increment the view count"""
        self.view_count += 1
        self.save()

    def __str__(self):
        return f"Page: {self.page_name}, Views: {self.view_count}"
