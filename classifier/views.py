from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from PIL import Image
import tensorflow as tf
import numpy as np
import os
from django.conf import settings

MODEL_PATH = os.path.join(settings.BASE_DIR, "cifar10_cnn_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def preprocess_image(image_file):
    img = Image.open(image_file).resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 32, 32, 3)
    return img_array

image_param = openapi.Parameter(
    'image',
    openapi.IN_FORM,
    description="Image file to classify",
    type=openapi.TYPE_FILE,
    required=True,
)

@swagger_auto_schema(method='post', manual_parameters=[image_param])
@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def predict(request):
    if "image" not in request.FILES:
        return Response({"error": "No image provided"}, status=400)

    img_array = preprocess_image(request.FILES["image"])
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    return Response({"prediction": predicted_class})
