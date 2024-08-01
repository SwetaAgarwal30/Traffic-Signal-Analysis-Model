import cv2
import numpy as np
import os
from tkinter import filedialog
from tkinter import *
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
import os

# Define the car color model architecture
def create_car_color_model():
    model = tf.keras.Sequential([
        Input(shape=(64, 64, 3)),  # Assuming the input shape is (64, 64, 3)
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')  # Assuming 2 classes: Red and Blue
    ])
    return model

# Define the car count model architecture
def create_car_count_model():
    model = tf.keras.Sequential([
        Input(shape=(64, 64, 3)),  # Assuming the input shape is (64, 64, 3)
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # Assuming regression for count
    ])
    return model

# Define the gender model architecture
def create_gender_model():
    model = tf.keras.Sequential([
        Input(shape=(64, 64, 1)),  # Assuming the input shape is (64, 64, 1) for grayscale images
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')  # Assuming 2 classes: Male and Female
    ])
    return model

# Function to load the gender model
def load_gender_model():
    proto_path = 'deploy_gender.prototxt'
    model_path = 'gender_net.caffemodel'
    if not os.path.exists(proto_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Model files not found: 'deploy_gender.prototxt' or 'gender_net.caffemodel'")
    return cv2.dnn.readNetFromCaffe(proto_path, model_path)

# Function to predict car color
def predict_car_color(model, image):
    image = cv2.resize(image, (64, 64))
    image = np.expand_dims(image, axis=0) / 255.0
    prediction = model.predict(image)
    if prediction[0][0] < 0.5:
        return 'blue'  # Assuming index 0 corresponds to blue
    else:
        return 'red'   # Assuming index 1 corresponds to red

# Function to count cars and people
def count_cars_and_people(image):
    car_cascade = cv2.CascadeClassifier('cars.xml')
    people_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect cars
    cars = car_cascade.detectMultiScale(gray_image, 1.1, 1)
    num_cars = len(cars)
    
    # Detect people
    people = people_cascade.detectMultiScale(gray_image, 1.1, 1)
    num_people = len(people)

    return num_cars, num_people

# Function to count males and females
def count_males_and_females(image):
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Load the gender detection model
    gender_net = load_gender_model()
    GENDER_LIST = ['Male', 'Female']
    
    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    num_males = 0
    num_females = 0
    
    for (x, y, w, h) in faces:
        face_img = image[y:y+h, x:x+w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = GENDER_LIST[gender_preds[0].argmax()]
        
        if gender == 'Male':
            num_males += 1
        else:
            num_females += 1
    
    return num_males, num_females

# GUI Setup
top = Tk()
top.geometry('800x600')
top.title('Traffic Signal Analysis')
top.configure(background="#CDCDCD")

label1 = Label(top, background="#CDCDCD", font=('arial', 15, 'bold'))
sign_image = Label(top)

car_color_model = create_car_color_model()
car_color_model.load_weights('car_color_model.weights.h5')
car_count_model = create_car_count_model()
car_count_model.load_weights('car_count_model.weights.h5')
gender_model = create_gender_model()
gender_model.load_weights('gender_model.weights.h5')

def detect(file_path):
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Predict car color
    car_color = predict_car_color(car_color_model, image)
    
    # Count cars and people
    num_cars, num_people = count_cars_and_people(image)
    
    # Count males and females
    num_males, num_females = count_males_and_females(image)
    
    result = f"Car Color: {car_color}, Cars: {num_cars}, People: {num_people}, Males: {num_males}, Females: {num_females}"
    label1.config(text=result)
    label1.pack()
    sign_image.pack()

upload = Button(top, text="Upload Image", command=lambda: detect(filedialog.askopenfilename()), padx=10, pady=5)
upload.pack(side=TOP, pady=50)

top.mainloop()
