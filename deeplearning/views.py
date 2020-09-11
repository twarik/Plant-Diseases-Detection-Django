from django.shortcuts import render
from .dl_model.model import classify_image
from django.core.files.storage import FileSystemStorage


# Create your views here.
def home(request):
    return render(request, 'deeplearning/index.html')

def about(request):
    return render(request, 'deeplearning/about.html')

def plant(request):
    return render(request, 'deeplearning/plant.html')

def predict(request):
    if request.method == "GET":
        return render(request, 'deeplearning/predict.html')
    if request.method == 'POST':
        # Access the input (image) stream and keep it in the Filestorage
        unploaded_file = request.FILES['myfile']
        #convert the file to bytes
        image = unploaded_file.read()
        # predict the class of the image
        result = classify_image(image)
        #Select the top three predictions according to their probabilities
        top1 = '1. Species: %s, Status: %s, Probability: %.4f'%(result[0][0], result[0][1], result[0][2])
        top2 = '2. Species: %s, Status: %s, Probability: %.4f'%(result[1][0], result[1][1], result[1][2])
        top3 = '3. Species: %s, Status: %s, Probability: %.4f'%(result[2][0], result[2][1], result[2][2])

        predictions = [ { 'pred':top1 }, { 'pred':top2 }, { 'pred':top3 } ]
        context = { 'predictions':predictions }

        # ## In addition to image classification, Let's store the predicted filecd
        # # Save the file to ./media
        fs = FileSystemStorage()
        filename = fs.save(unploaded_file.name, unploaded_file)
        uploaded_file_url = fs.url(filename)
        context['url'] = uploaded_file_url

        return render(request, 'deeplearning/predict.html', context)
