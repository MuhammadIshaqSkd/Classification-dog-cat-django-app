from django.shortcuts import render
from django.http import JsonResponse
from imageprogress.ml_model import predict_image

# Create your views here.
def predict(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']  # This is the file-like object (InMemoryUploadedFile)
        try:
            # Call the prediction function, passing the image file (not a file path)
            label, probabilities = predict_image(image_file)

            # Check if probabilities is a 2D array or scalar and handle accordingly
            if probabilities.ndim == 1:
                probabilities = {'cats': float(probabilities[0]), 'dogs': float(probabilities[1])}
            else:
                probabilities = {'cats': float(probabilities[0][0]), 'dogs': float(probabilities[0][1])}

            # Return the prediction and probabilities as a JSON response
            return JsonResponse({
                'label': label,
                'probabilities': probabilities
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return render(request, 'predict_image.html')
