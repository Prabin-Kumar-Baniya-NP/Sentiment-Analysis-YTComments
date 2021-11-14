from django.http.response import HttpResponse, JsonResponse
from django.shortcuts import render
from ytcomments.modules.sentimentanalysis import analysizeComments
from ytcomments.modules.comments import commentExtract

# Create your views here.
def index(request):
    return render(request, "ytcomments/index.html", {})

def analysis(request):
    videoURL = str(request.GET.get("videoURL"))
    videoId = videoURL[videoURL.find("v=")+2: videoURL.find("&")]
    result = analysizeComments(videoId, 2000)
    return render(request, "ytcomments/result.html", {"result": result})