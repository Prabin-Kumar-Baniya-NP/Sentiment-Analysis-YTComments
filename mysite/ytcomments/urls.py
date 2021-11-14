from django.urls import path
from ytcomments import views

app_name = "ytcomments"

urlpatterns = [
    path("", views.index, name="index-view"),
    path("comments-analysis/", views.analysis, name="comments-analysis-result")
]
