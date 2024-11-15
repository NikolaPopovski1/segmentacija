import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def iskalnik_4_kotnikov(slika: np.ndarray) -> list[np.ndarray]:
    # Pretvori v sivinsko sliko z povprečenjem RGB kanalov
    #slika = slika.mean(2)

    #img = cv.imread('test.png', cv.IMREAD_GRAYSCALE)
    # Uporabi gaussov filter iz knjižnice ndimage

    img = cv.GaussianBlur(slika, (3, 3), 2)

    edges = cv.Canny(img, 100, 10)

    contours, _ = cv.findContours(edges, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)

    # Združi bližnje konture
    combined_contours = []
    for contour in contours:
        if len(combined_contours) == 0:
            combined_contours.append(contour)
        else:
            merged = False
            for i, combined_contour in enumerate(combined_contours):
                for point in contour:
                    distance = cv.pointPolygonTest(combined_contour, (int(point[0][0]), int(point[0][1])), True)
                    if distance >= 0 and distance < 10:  # Prag razdalje za združevanje kontur
                        combined_contours[i] = np.vstack((combined_contour, contour))
                        merged = True
                        break
                if merged:
                    break
            if not merged:
                combined_contours.append(contour)

    # Uporabi konveksni ovoj za združevanje kontur
    hulls = [cv.convexHull(contour) for contour in combined_contours]

    # Aproksimiraj združene konture in najdi pravokotnike
    rectangles = []
    for contour in hulls:
        # Aproksimiraj konturo
        epsilon = 0.015 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)
        # Preveri, če ima aproksimirana kontura 4 oglišča
        if len(approx) == 4:
            rectangles.append(approx)

    #"""Prikaz rezultatov
    # Prikaži rezultate
    plt.imshow(img, cmap='gray')
    plt.title('Originalna slika'), plt.xticks([]), plt.yticks([])
    #plt.subplot(122), plt.imshow(edges, cmap='gray')
    #plt.title('Slika robov'), plt.xticks([]), plt.yticks([])

    for rect in rectangles:
        #print("Rectangle:")
        #print(rect)
        plt.plot(rect[:, 0, 0], rect[:, 0, 1], 'r')
        plt.plot([rect[0, 0, 0], rect[-1, 0, 0]], [rect[0, 0, 1], rect[-1, 0, 1]], 'r')

    plt.show()
    #"""

    return rectangles