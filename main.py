# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import cv2
import matplotlib.pyplot as plt


def rotation(image, theta):
    '''
    not consider theta < 0 or theta > 45 (that is k < -45)
    :param image:
    :param theta:
    :return:
    '''
    # return_image = np.zeros(shape=image.shape)
    theta = theta % (2 * np.pi)
    cosine = np.cos(theta)
    sine = np.sin(theta)
    affine_matrix = np.array([[cosine,sine,0],[-sine,cosine,0],[0,0,1]])
    a = len(image)
    b = len(image[0])
    pad_length = a*cosine
    return_image = np.zeros(shape=(int(a*cosine + b*sine),int(a*sine + b*cosine)))
    for i in range(a):
        for j in range(b):
            test = np.array([[i],[j],[image[i][j]]])
            index = np.matmul(affine_matrix,test)
            try:
                return_image[int(index[0][0])][int(index[1][0] + a * sine)] = index[2]
            except IndexError:
                print(a)
                print(b)
    # return_image = convolution(affine_matrix, image)

    return return_image


def derivative(image,k):
    '''

    :param image:
    :param k: relative to the positive direction
    :return:
    '''
    # filter = np.array([[-1,1-np.tan(k)],[0,np.tan(k)]])
    filter = np.array([[-1,1-np.tan(k)],[0,np.tan(k)]])
    length = 1
    temp_image = np.pad(image,((length,length),(length,length)),'constant',constant_values = (0,0))
    intensity_image = np.zeros(shape = temp_image.shape)
    for i in range(length,len(temp_image)):
        for j in range(length,len(temp_image[0])):
            value = np.sum(np.multiply(filter,temp_image[i - length:i + length , j - length:j + length]))
            intensity_image[i][j] = value

    return intensity_image


def findDirection(image):
    intensity_list = np.zeros(45)
    for k in range(45):
        intensity_image = derivative(image, - k/180 * (np.pi))
        # plt.imshow(intensity_image, cmap='gray')
        # plt.show()
        intensity_value = np.abs(np.sum(np.sum(intensity_image)))# slightly different from origin, and a bit of slow
        intensity_list[k] = intensity_value
    ind = np.argmax(intensity_list)
    return intensity_list


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path = 'test2.tiff'
    image = cv2.imread(path, 0)
    # output_image = rotation(image, np.pi/2)
    # print(output_image)
    print(findDirection(image))
    # output_image = derivative(image,0)
    # plt.imshow(output_image, cmap='gray')
    # plt.show()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
