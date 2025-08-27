# Midterms: CIFAR-10 Image Classification with Django API

This project demonstrates how to train a Convolutional Neural Network (CNN) using the **CIFAR-10 dataset** and then serve predictions via a **Django REST API**. Students will train the model, save it, and create an endpoint where users can upload an image (via Postman or similar tool), and the API will respond with the predicted class.

---

## Learning Objectives
- Learn how to work with another built-in dataset in TensorFlow (CIFAR-10).
- Train a CNN for image classification.
- Save and load trained models.
- Integrate a TensorFlow model with a Django backend.
- Send image data via API (Postman) and get predictions in JSON response.

---

## 📌 Overview
In this activity, you will:
1. Train a Convolutional Neural Network (CNN) using the **CIFAR-10 dataset** (a built-in dataset in TensorFlow).
2. Save the trained model for later use.
3. Build a **Django REST API** that accepts an image (via Postman) and returns the predicted class (e.g., airplane, cat, ship, etc.).

This is an extension of the MNIST activity. Instead of digits, you will now classify **real-world images**.

---

## 📂 Dataset: CIFAR-10
CIFAR-10 is a dataset of **60,000 32x32 color images** in **10 classes**:
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

Each class has **6,000 images**. The dataset is built into TensorFlow, so you don’t need to download it manually.

---

## Requirements
Make sure you have the following installed:
- Python 3.9+
- pip (Python package manager)
- Virtual environment (recommended)

Install dependencies:

```bash
pip install tensorflow pillow numpy django djangorestframework
```

---

## 📊 Dataset
We’ll use the **CIFAR-10 dataset**, which contains 60,000 images across **10 classes**:
- airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

TensorFlow can load it directly:

```python
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

---

## 🏋️ Training the Model
Train and save your model as `cifar10_cnn_model.h5`:

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

# Load dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# Compile and train
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Save model
model.save("cifar10_cnn_model.h5")
```

---

## 🌐 Django API Setup
1. Create a Django project:
```bash
django-admin startproject cifar_api
cd cifar_api
python manage.py startapp classifier
```

2. Add `rest_framework` and `classifier` to `INSTALLED_APPS` in `settings.py`.

3. In `classifier/views.py`, create an endpoint for prediction:

```python
import tensorflow as tf
import numpy as np
from rest_framework.decorators import api_view
from rest_framework.response import Response
from PIL import Image

# Load model once
model = tf.keras.models.load_model("cifar10_cnn_model.h5")
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def preprocess_image(image_file):
    img = Image.open(image_file).resize((32,32))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1,32,32,3)
    return img_array

@api_view(["POST"])
def predict(request):
    if "image" not in request.FILES:
        return Response({"error": "No image provided"}, status=400)

    img_array = preprocess_image(request.FILES["image"])
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    return Response({"prediction": predicted_class})
```

4. Add to `classifier/urls.py`:

```python
from django.urls import path
from .views import predict

urlpatterns = [
    path("predict/", predict, name="predict"),
]
```

5. Include in main `urls.py`:

```python
from django.urls import path, include

urlpatterns = [
    path("api/", include("classifier.urls")),
]
```

---

## ▶️ Running the Server
Run the server:
```bash
python manage.py runserver
```

---

## 📬 Testing with Postman
1. Open Postman.
2. Send a `POST` request to:
   ```http
   http://127.0.0.1:8000/api/predict/
   ```
3. In the **Body**, choose `form-data` and upload a key named `image` with a `.png` or `.jpg` file.
4. You should receive a response like:
```json
{
  "prediction": "dog"
}
```

---

## 📝 What to Submit

You must submit the following via your LMS or upload system:

1. ✅ A **PDF report** named `SOFTDSNL_Midterms_Surnames.pdf` that includes:
   - Screenshots of model training results (accuracy/loss).
   - Screenshots of Postman prediction results (10 screenshots, 1 for each category)
   - **Link to your GitHub repository fork**

---

