import numpy as np
import random
import cv2
import os

def sp_noise(image, prob):
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

counter = 0
directory = r'X:\dataset\img'
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        print(os.path.join(directory, filename))
        image = cv2.imread(os.path.join(directory, filename), 0)  # Only for grayscale image
        noise_img = sp_noise(image, 0.05)
        cv2.imwrite('X:/dataset/noised/'+ str(counter) +'.jpg', noise_img)
        cv2.imwrite('X:/dataset/grey/'+ str(counter) + '.jpg', image)
    else:
        continue
    counter += 1

#source https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv