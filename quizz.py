import cv2
import tensorflow as tf
import numpy as np
from tkinter import filedialog
from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image, ImageOps

top=tk.Tk()
top.geometry('800x600')
top.title('Traffic Light image classification')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
output_image = Label(top)

# Load the pre-trained model
model = tf.keras.models.load_model('model_hand.h5')
# Define a function to preprocess the input image
def preprocess_image(image_path):
    # Load the image and resize it to the input size expected by the model
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (20, 100))  # Resize to match the model's input size
    image = image / 255.0  # Normalize the image to [0, 1]
    return image

# Define a function to classify the traffic light color
def classify_traffic_light(image_path):

    disp_string= ''
    # Preprocess the image
    image = preprocess_image(image_path)

    # Make predictions using the pre-trained model
    predictions = model.predict(np.array([image]))

    # Get the class with the highest probability
    predicted_class = np.argmax(predictions)

    # Define the class labels (assuming the model output is [red, yellow, green])
    class_labels = ["Back", "Red", "Yellow", "Green"]

    # Get the predicted class label
    predicted_label = class_labels[predicted_class]

    disp_string += "\n Color:"+str(predicted_label)
    label.configure(foreground='#011638', text=disp_string)

def show_classify_button(image_path):
    classify_b=Button(top,text="Classify Image",command=lambda: classify_traffic_light(image_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)
        output_image.configure(image=im)
        output_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass



upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=50)
output_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Traffic Light Image Classifier",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()

