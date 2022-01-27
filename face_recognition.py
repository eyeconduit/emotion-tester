import numpy as np

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import load_model

def emotion_detection(image_path):
    train_datagen = ImageDataGenerator(
        zoom_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
        rescale=1./255
    )

    train_data = train_datagen.flow_from_directory(directory="train", target_size=(224, 224), batch_size=32)

    model = load_model('model_data.h5')

    op = dict(zip(train_data.class_indices.values(), train_data.class_indices.keys()))

    path = f'static/uploads/{image_path}'
    img = load_img(path, target_size=(224, 224))

    i = img_to_array(img)/255
    input_arr = np.array([i])

    pred = np.argmax(model.predict(input_arr))
    return f"This image is {op[pred]}"

    # display image
    # cv2.imshow('input image', input_arr[0])
    # cv2.waitKey(0)




# import cv2
# from keras.models import load_model
#
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# # cv2.namedWindow("imgz", cv2.WINDOW_NORMAL)
# # img = cv2.imread('demo.jpg')
# # img_resize = cv2.resize(img, (1060, 540))
# img = cv2.imread('test_front.jpg')
# img_resize = cv2.resize(img, (224, 224))
#
# gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
#
# faces = face_cascade.detectMultiScale(gray, 1.1, 4)
#
# for (x, y, w, h) in faces:
#     cv2.rectangle(img_resize, (x, y), (x+w, y+h), (255, 0, 0), 2)
#
# cv2.imshow('img', img_resize)
# cv2.waitKey(0)