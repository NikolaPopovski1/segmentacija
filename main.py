import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import iskalniki

image_path = os.path.abspath('test.png')
slika = cv.imread(image_path)

rezultat = iskalniki.iskalnik_4_kotnikov(slika)

"""Show rectangles
for rect in rezultat:
        #print("Rectangle:")
        #print(rect)
        plt.plot(rect[:, 0, 0], rect[:, 0, 1], 'r')
        plt.plot([rect[0, 0, 0], rect[-1, 0, 0]], [rect[0, 0, 1], rect[-1, 0, 1]], 'r')

#plt.show()

""""""Regular Img Show
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(slika, cmap='gray')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.show()
"""