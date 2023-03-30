# import opencv library and print the version
# a version above 3.0 is recommended
import cv2


print("OpenCV Version: ", cv2.__version__)
# import numpy library
import numpy as np
# import matplotlib charting library
import matplotlib.pyplot as plt
# Load a JPG image as an array
my_image = cv2.imread('data/monalisa.jpg')
# convert the image from BGR to RGB color space
my_image = cv2.cvtColor(my_image, cv2.COLOR_BGR2RGB)
# Show size of the array
print("Original image array shape: ", my_image.shape)
# Show values for pixel (100,100)
print ("Pixel (100,100) values: ", my_image[100][100][:])
# Resize the image
my_image = cv2.resize(my_image, (400,600))
# plt.imshow(my_image)
# plt.show()
# Show size of the array
print("Resized image array shape: ", my_image.shape)
# convert the image from RGB to BGR color space
my_image = cv2.cvtColor(my_image, cv2.COLOR_RGB2BGR)
# Save the new image
cv2.imwrite('data/new_monalisa.jpg', my_image)
# convert the image to greyscale
my_grey = cv2.cvtColor(my_image, cv2.COLOR_RGB2GRAY)
print('Image converted to grayscale.')
# plt.imshow(my_grey, cmap='gray')
# plt.show()


my_image = cv2.imread('data/new_monalisa.jpg')
my_image = cv2.cvtColor(my_image, cv2.COLOR_BGR2RGB)

# my_image[10:100,10:100,:] = 0
# plt.imshow(my_image)
#
# my_image[10:100,300:390,:] = 0
# my_image[10:100,300:390,0] = 255
# plt.imshow(my_image)
#
# roi = my_image[50:250,125:250,:]
#
# roi = cv2.resize(roi, (300,300))
# my_image[300:600,50:350,:] = roi
# plt.imshow(my_image)
# plt.show()

def show_image(p_image,p_title):
    plt.figure(figsize=(5,10))
    plt.axis('off')
    plt.title(p_title)
    plt.imshow(p_image)

temp_image = my_image.copy()

# cv2.line(temp_image, (10,100), (390,100), (0,255,255), 5)
#
# cv2.rectangle(temp_image, (200,200), (300,400), (0,255,255), 5)
#
# cv2.circle(temp_image, (100,200), 50, (255,0,0), -1)
#
# font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(temp_image, 'Mona Lisa', (10,500), font, 1.5, (255,255,255), 2, cv2.LINE_AA)
#
# show_image(temp_image, 'Result 1: Draw geometry and text')
# plt.show()

# gray = cv2.cvtColor(temp_image, cv2.COLOR_RGB2GRAY)
#
# ret, thresh1 = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
# ret, thresh2 = cv2.threshold(gray,100,255,cv2.THRESH_BINARY_INV)
# ret, thresh3 = cv2.threshold(gray,100,255,cv2.THRESH_TRUNC)
# ret, thresh4 = cv2.threshold(gray,100,255,cv2.THRESH_TOZERO)
# ret, thresh5 = cv2.threshold(gray,100,255,cv2.THRESH_TOZERO_INV)
#
# titles = ['Original Image', 'BINARY Threshold','BINARY_INV Threshold', 'TRUNC Threshold', 'TOZERO Threshold','TOZERO_INV Threshold']
#
# images = [gray, thresh1, thresh2, thresh3, thresh4, thresh5]
#
# plt.figure(figsize=(15,15))
# for i in np.arange(6):
#     plt.subplot(2,3,i+1),plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.axis('off')
#
# plt.show()



# show_image(temp_image, 'Original Image')
#
# kernel = np.ones((3,3),np.float32)/9
# result = cv2.filter2D(temp_image,-1,kernel)
#
# result = cv2.filter2D(result,-1,kernel)
# result = cv2.filter2D(result,-1,kernel)
#
# show_image(result, 'Result: Blurring filter')
#
# kernel_sharpening = np.array([[-1,-1,-1],
#                              [-1,9,-1],
#                              [-1,-1,-1]])
# result = cv2.filter2D(temp_image,-1,kernel_sharpening)
# show_image(result,'Result: Sharpening filter')
#
# plt.show()

#
# gray = cv2.cvtColor(temp_image,cv2.COLOR_RGB2GRAY)
# edges = cv2.Canny(gray,100,255)
#
# plt.figure(figsize=(5,10))
# plt.axis('off')
# plt.title('Result: Canny Edge detection')
# plt.imshow(edges, cmap='gray')
# plt.show()






# haar cascades

gray = cv2.cvtColor(temp_image, cv2.COLOR_RGB2GRAY)
face_cascade = cv2.CascadeClassifier('data/haarcascade_profileface.xml')
faces = face_cascade.detectMultiScale(gray,1.3,5)
for (x,y,w,h) in faces:
    roi_color = temp_image[y:y+h, x:x+w]
    show_image(roi_color, 'Result: ROI of face detected by Haar Cascade Classifier')
    cv2.rectangle(temp_image, (x,y), (x+w,y+h), (0,255,0),2)

show_image(temp_image, 'Result: Face detection using Haar Cascade Classifier')
plt.show()