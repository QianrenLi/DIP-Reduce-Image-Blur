# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import cv2
import matplotlib.pyplot as plt


def convolution(affine_matrix, input_image):
    size = len(affine_matrix)
    length = size // 2
    n = len(input_image)
    m = len(input_image[0])
    temp_image = np.pad(input_image,((n,m),(n,m)),'constant',constant_values = (0,0))

    return_image = np.zeros(shape=temp_image.shape)
    for i in range(n+length, 2*n):
        for j in range(m+length, 2*m):
            test = np.array([[i],[j],[temp_image[i][j]]])
            index = np.matmul(affine_matrix,test)
            a = int(index[0][0])
            b = int(index[1][0])
            # temp = temp_image[i - length:i + length + 1, j - length: j + length + 1]
            # print(a.dtype)
            try:
                return_image[a][b] = index[2][0]
            except IndexError:
                print(a)
                print(b)
    return return_image


def rotation(image, theta):
    # return_image = np.zeros(shape=image.shape)
    affine_matrix = np.array([[np.cos(theta),np.sin(theta),0],[-np.sin(theta),np.cos(theta),0],[0,0,1]])
    return_image = convolution(affine_matrix, image)

    return return_image


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path = 'test.tiff'
    image = cv2.imread(path, 0)
    output_image = rotation(image, 0.5)
    # print(output_image)
    plt.imshow(output_image, cmap='gray')
    plt.show()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
