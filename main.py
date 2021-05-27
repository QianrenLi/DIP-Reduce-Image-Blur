# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.signal as signal
from PIL import Image as im


def _blur_image_horizontal_test(image, extent):
    new_image = np.zeros(shape=(50, 50))
    length = 40
    for i in range(length, length + len(new_image)):
        for j in range(length, length + len(new_image[0])):
            index = j - extent if j - extent >= 0 else 0
            value = np.sum(image[i][index: j])
            value_next = value + float(image[i][j])
            new_image[i - length][j - length] = value_next
    return new_image


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
    affine_matrix = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
    a = len(image)
    b = len(image[0])
    pad_length = a * cosine
    return_image = np.zeros(shape=(int(np.ceil(a * cosine + b * sine)), int(np.ceil(a * sine + b * cosine))))
    for i in range(a):
        for j in range(b):
            test = np.array([[i], [j], [image[i][j]]])
            index = np.matmul(affine_matrix, test)
            try:
                return_image[int(index[0][0] + b * sine)][int(index[1][0])] = index[2]
            except IndexError:
                # pass
                print(a)
                print(b)
    # another version:
    # np.array([[cosine,sine,0],[sine,-cosine,0],[0,0,1]]);
    # return_image[int(index[0][0])][int(index[1][0] - a * cosine)] = index[2] ?
    # return_image = convolution(affine_matrix, image)

    return return_image


# def wiener_filter_in_time_domain()


def mid_filter(input_image, size):
    '''
    useful in filtering the salt and pepper noise
    :param input_image:
    :param size: the range of neighbors, odd number is recommend
    :return: 2-D matrix with less noise
    '''
    length = size // 2
    N = len(input_image)
    M = len(input_image[0])
    out_image = np.empty(shape=(N + length * 2, M + length * 2), dtype=float)
    output_image = np.empty(shape=(N, M), dtype=float)
    for i in range(length, N + length):
        for j in range(length, M + length):
            out_image[i][j] = input_image[i - length][j - length]
    midian = (size * size) // 2
    for i in range(length, N + length):
        for j in range(length, M + length):
            local_array = out_image[i - length:i + length + 1, j - length:j + length + 1]
            local_array = [element for elements in local_array for element in elements]
            # print(local_array)
            local_array.sort()
            temp = int((local_array[midian] / 2 + local_array[~midian] / 2))
            output_image[i - length][j - length] = temp

    output_image = output_image.astype(np.uint8)
    return output_image


def derivative(image, k):
    '''

    :param image:
    :param k: relative to the positive direction
    :return:
    '''
    # filter = np.array([[-1,1-np.tan(k)],[0,np.tan(k)]])
    tan_coefficient = np.tan(k)
    print(tan_coefficient)
    length = 1
    temp_image = np.pad(image, ((length, length), (length, length)), 'constant', constant_values=(0, 0))
    temp_image = temp_image.astype(float)
    intensity_image = np.zeros(shape=temp_image.shape)
    for i in range(length, len(temp_image) - 2 * length):
        for j in range(length, len(temp_image[0]) - 2 * length):
            value = temp_image[i][j + 1] * (1 - tan_coefficient) + temp_image[i + 1][j + 1] * tan_coefficient - \
                    temp_image[i][j]
            intensity_image[i][j] = value

    return intensity_image


# def goldenSearch(image):
#     pass


def findDirection(image):
    intensity_list = np.zeros(45)
    for k in range(18):
        intensity_image = derivative(image, k / 36 * (2 * np.pi))  # k 其实为 -k
        # plt.imshow(intensity_image, cmap='gray')
        # plt.show()
        intensity_value = np.sum(np.sum(np.abs((intensity_image))))  # slightly different from origin, and a bit of slow
        intensity_list[k] = intensity_value

    return intensity_list


def computeACF(rotatedImage, times, length) -> object:  # here the rotatedImage is the image after derivative

    # for k in range(len(intensity_image)):
    # k = int(len(rotatedImage)/2)
    autocorrelation = np.zeros(shape=(2 * length + 1), dtype=float)
    # for k in range(len(intensity_image)):
    # k = int(len(rotatedImage)/2)
    # times = 10
    for k in range(0, len(rotatedImage), times):
        for j in range(2 * length + 1):
            for i in range(2 * length + 1):
                ref_value_test = rotatedImage[k][i - length] if (2 * length > i >= length) else 0
                ref_value = rotatedImage[k][(i + j) - 2 * length] if (3 * length > i + j >= 2 * length) else 0
                autocorrelation[j] += ref_value_test * ref_value

    max_index = np.argmax(autocorrelation)
    min_index = np.argmin(autocorrelation)
    intensity = np.abs(max_index - min_index)
    print(np.argmin(autocorrelation))
    print(np.argmax(autocorrelation))
    autocorrelation = autocorrelation / int(len(rotatedImage) / times)
    # autocorrelation = autocorrelation / (max(autocorrelation) / (max_ind-min_ind))
    autocorrelation = autocorrelation / (max(autocorrelation) / np.power((max_index - min_index), 2))

    return (autocorrelation, intensity)


def integral_by_slicing(MTF):  # 0 ~ 2pi ? -> shift may need
    half_length = len(MTF) // 2
    # shifted_MTF = np.zeros(shape=MTF.shape, dtype=complex)
    # shifted_MTF[half_length:-1] = MTF[0: half_length]
    # shifted_MTF[0:half_length] = MTF[half_length:-1]
    PTF = np.zeros(shape=MTF.shape, dtype=float)
    k = len(MTF)
    slice_size = 2 * np.pi / k
    # for u in range(len(PTF)):
    #     u_slice = u / k * 2 * np.pi
    #     for alpha in range(k):
    #         alpha_slice = alpha / k * 2 * np.pi
    #         PTF[u] += np.log(MTF(alpha)) / np.tan(
    #             (u_slice - alpha_slice) / 2) * slice_size  # Hilbert transform list in the paper
    # PTF = - signal.hilbert(np.log(MTF)) # high resolution image
    PTF = - signal.hilbert(np.log10(MTF))  # low pixel image
    return PTF


def computeMTF(autocorrelation):
    SdPSF = np.fft.fftshift(np.fft.fft(autocorrelation))
    length = len(SdPSF)
    Du = np.ones(shape=(2 * length + 1), dtype=float)
    for i in range(2 * length + 1):
        Du[i] = abs(2 * np.pi / 2 / length * (i - length))

    MTFu = np.divide(np.sqrt(np.abs(SdPSF)), Du)
    PTFu = np.ones(shape=(2 * length + 1), dtype=float)
    # for i in range(2*length + 1):
    #     for j in range(6):
    #         PTFu[i] += np.log(MTFu[j])/np.tan((i- length - j)/2)
    #     PTFu[i] *= - 1 / 2 / np.pi
    return (MTFu, PTFu)


def fromPowerDensityToSignal(SdPSF):
    length = len(SdPSF)
    signal = np.random.randn(1, length) / np.sqrt(length)
    plt.plot(signal)
    plt.show()
    Xk = np.fft.fft(signal) * abs(SdPSF)
    Xm = np.fft.ifft(Xk)
    xm = np.real(Xm)
    plt.plot(xm)
    plt.show()
    return xm


def normalization(input_image):
    max_num = np.max(input_image)
    min_num = np.min(input_image)
    # print(max_num)
    # print(min_num)
    #
    input_image = 255 * ((input_image - min_num) / (max_num - min_num))
    input_image = input_image.astype(np.uint8)
    return input_image


def MMSE_filter(input_image, H, sigma):
    output_image = np.zeros(shape=input_image.shape)
    lengthN = len(input_image)
    lengthM = len(input_image[0])
    input_image = np.fft.fft2(input_image)
    H = np.fft.fft2(H)
    for i in range(lengthN):
        for j in range(lengthM):
            # if distance < distance_max:
            a = np.power(np.abs(H[i][j]), 2)
            # if filter[i][j] > 0.0000000001:
            output_image[i][j] = input_image[i][j] / (H[i][j] + sigma / a)
    return np.fft.ifft2(output_image)


def generate_MTFandPTF(autocorrelation, length):
    SdPSF = np.fft.fft(autocorrelation)  # 2pi/length(SdPSF) * (index - length(PSF/2))
    MTF = np.sqrt(np.abs(SdPSF[0:length + 1]))
    Du = np.zeros(length + 1)
    Du[0] = 1
    for i in range(length):
        Du[i + 1] = np.pi / length * (i + 1)
    #
    # MTF = MTF / Du

    PTF = np.real(integral_by_slicing(MTF))
    MTF_full = np.append(MTF[length:1:-1], MTF)
    PTF_full = np.append(-PTF[length:1:-1], PTF)
    OSF = np.fft.ifft(np.fft.ifftshift(MTF_full * np.exp(1j * PTF_full)))
    return (MTF_full, PTF_full, OSF)


# Press the green button in the gutter to run the script.


def smoothingOSF(OSF, intensity):
    blur_function = OSF[0: intensity * 2]
    # blur_function = blur_function[::-1]
    # print(blur_function)
    smoothed_blur = np.zeros(shape=blur_function.shape)
    window_size = 5  # 2
    for i in range(len(smoothed_blur)):
        if i < window_size:
            for j in range(i):
                smoothed_blur[i] = smoothed_blur[i] + blur_function[j]
            smoothed_blur[i] = smoothed_blur[i] / (i + 1)
        else:
            for j in range(window_size):
                smoothed_blur[i] = smoothed_blur[i] + blur_function[i - j]
            smoothed_blur[i] = smoothed_blur[i] / window_size

    actual_blur = smoothed_blur
    actual_blur = actual_blur[1:intensity + 1]
    actual_blur = actual_blur / max(actual_blur)
    return (smoothed_blur, actual_blur)


def recoverImage(rotated_Image, actual_blur):
    length_temp_col = len(rotated_Image)
    length_temp_row = len(rotated_Image[0])
    rotated_Image = np.pad(rotated_Image, ((0, length_temp_col), (0, length_temp_row)), 'constant',
                           constant_values=(0, 0))
    padding = np.zeros(shape=rotated_Image.shape)
    # padding2 = np.zeros(int(len(rotated_Image[0]))//2)
    # print(padding[0,0:len(actual_blur)])
    padding[0, 0:len(actual_blur)] = padding[0, 0:len(actual_blur)] + actual_blur[0:len(actual_blur)]

    result_image = MMSE_filter(rotated_Image, padding, 10)
    return result_image


def hist_equ_11812417(input_image):
    L = 256
    input_hist = np.zeros((1, L))
    output_hist = np.zeros((1, L))
    M = len(input_image)
    N = len(input_image[0])
    number = M * N
    # print(M)
    # print(max(input_image[0]))
    for i in range(M):
        for j in range(N):
            # print(i)
            input_hist[0][input_image[i][j]] += 1

    input_hist = input_hist / number
    cdf = [input_hist[0][0]]
    for i in range(len(input_hist[0]) - 1):
        cdf.append(cdf[i] + input_hist[0][i + 1])

    cdf = np.array(cdf)
    # print(cdf)
    cdf = np.floor(cdf * (L - 1))  # floor?ceil?round
    # print(cdf)
    output_image = np.empty(shape=(M, N), dtype=np.uint8)

    for i in range(M):
        for j in range(N):
            output_image[i][j] = cdf[input_image[i][j]]
            output_hist[0][output_image[i][j]] += 1

    # output_hist

    # output_image = im.fromarray(output_image)
    # may need type convention
    input_hist = (input_hist[0] * number).tolist()
    output_hist = output_hist[0].tolist()
    # print(output_image)
    # print(output_hist)

    # plt.plot(output_hist)
    # print(input_image)
    # print(output_image)
    # plt.show()
    return output_image, output_hist, input_hist


if __name__ == '__main__':
    path = 'test3.tiff'
    # path = 'test7.png'
    image = cv2.imread(path, 0)

    # # rotation test
    # rotated_image = rotation(image,4.5/36*(2*np.pi))
    # plt.imshow(rotated_image)
    # plt.show()

    # test_image = np.ones(shape=(100,100))
    # test_image[55][60] = 0
    # test_image[60][55] = 0
    # test_image[70][57] = 0
    # test_image = normalization(test_image)
    # blur_image = _blur_image_horizontal_test(test_image,12)
    # image_final = im.fromarray(normalization(blur_image))
    # image_final.save("test4.tiff")
    # plt.imshow(test_image)
    # plt.imshow(image_final)
    # # # plt.savefig("test3.tiff")
    # plt.show()

    plt.imshow(image, cmap='gray')
    plt.show()
    min_ind = 0
    # image = mid_filter(image,3)
    intensity_image = derivative(image, min_ind / 36 * (2 * np.pi))
    # rotatedImage = intensity_image
    rotatedImage = rotation(intensity_image, min_ind / 36 * (2 * np.pi))
    # # for i in range(len(rotatedImage[0])):
    # #     rotatedImage[5][i] = 255

    length = len(rotatedImage[0])
    # plt.imshow(rotatedImage, cmap='gray')
    # plt.show()
    #
    # autocorrelation = np.zeros(shape=(2 * length + 1), dtype=float)
    # # for k in range(len(intensity_image)):
    # # k = int(len(rotatedImage)/2)
    # times = 1
    # for k in range(0, len(rotatedImage), times):
    #     for j in range(2 * length + 1):
    #         for i in range(2 * length + 1):
    #             ref_value_test = rotatedImage[k][i - length] if (2 * length > i >= length) else 0
    #             ref_value = rotatedImage[k][(i + j) - 2 * length] if (3 * length > i + j >= 2 * length) else 0
    #             autocorrelation[j] += ref_value_test * ref_value
    #
    # max_index = np.argmax(autocorrelation)
    # min_index = np.argmin(autocorrelation)
    # intensity = np.abs(max_index - min_index)
    # print(np.argmin(autocorrelation))
    # print(np.argmax(autocorrelation))
    # autocorrelation = autocorrelation / int(len(rotatedImage) / times)
    # # autocorrelation = autocorrelation / (max(autocorrelation) / (max_ind-min_ind))
    # autocorrelation = autocorrelation / (max(autocorrelation) / np.power((max_index - min_index),2))

    (autocorrelation, intensity) = computeACF(rotatedImage, 1, length)

    plt.plot(autocorrelation)
    plt.show()

    # SdPSF = np.fft.fft(autocorrelation)  # 2pi/length(SdPSF) * (index - length(PSF/2))
    # MTF = np.sqrt(np.abs(SdPSF[0:length + 1]))
    # Du = np.zeros(length + 1)
    # Du[0] = 1
    # for i in range(length):
    #     Du[i + 1] = np.pi / length * (i + 1)
    # #
    # # MTF = MTF / Du
    #
    # PTF = np.real(integral_by_slicing(MTF))
    # MTF_full = np.append(MTF[length:1:-1], MTF)
    # PTF_full = np.append(-PTF[length:1:-1], PTF)
    (MTF_full,PTF_full,OSF) = generate_MTFandPTF(autocorrelation,length)
    plt.plot(MTF_full)
    plt.show()
    # fromPowerDensityToSignal(SdPSF)
    # length = len(SdPSF)
    # loop_time = 10000
    # xm = np.zeros(length)
    # # for k in range(loop_time):
    # #     signal = np.random.randn(length)
    # #     # print(signal)
    # #     # plt.plot(signal)
    # #     # plt.show()
    # #     # print(abs(SdPSF))
    # #     Xk = np.fft.fftshift(np.fft.fft(signal)) * abs(SdPSF)
    # #     Xm = np.fft.ifft(np.fft.ifftshift(Xk))
    # #     xm += np.real(Xm)
    #     # print(xm)
    # signal = np.random.randn(length)
    # plt.plot(xm/loop_time)
    # plt.show()
    # MTF = np.sqrt(np.abs(SdPSF))
    # OSF = np.fft.ifft(SdPSF)
    # plt.plot(np.real(SdPSF))
    # plt.show()
    # SdPSF = np.fft.fft(autocorrelation)
    # MTF = autocorrelation
    # MTF = np.ones(shape=(2 * length + 1), dtype=float)
    # for i in range(2 * length + 1):
    #     # Du = abs(2 * np.pi / 2 * (i - length))
    #     Du = abs((i - length))
    #     if i == length:
    #         MTF[i] = 1
    #     else:
    #         MTF[i] = np.sqrt(np.abs(SdPSF[i])) / Du

    # MTF = np.fft.ifftshift(MTF)
    # shifted_MTF = np.zeros(shape=MTF.shape,dtype = complex)
    # shifted_MTF[half_length:-1] = MTF[0: half_length]
    # shifted_MTF[0:half_length] = MTF[half_length:-1]
    #
    # half_MTF = MTF[0:length]
    # PTF = np.zeros(shape=(2 * length + 1), dtype=float)
    # PTF = integral_by_slicing(MTF)
    # half_PTF = PTF[0:length]
    # # OSF = np.fft.ifft(np.fft.ifftshift(half_MTF*np.exp(1j * np.real(half_PTF))))
    # OSF = np.fft.ifft(np.fft.ifftshift(MTF ))
    OSF = np.fft.ifft(np.fft.ifftshift(MTF_full * np.exp(1j * PTF_full)))

    # # OSF = np.fft.ifft(np.fft.ifftshift(SdPSF))
    # plt.plot(MTF)
    # plt.show()
    # plt.plot(np.real(PTF))
    # plt.show()
    OSF = np.real(OSF)
    plt.plot(np.real(OSF))
    plt.show()

    #
    blur_function = OSF[0: intensity * 2]
    # blur_function = blur_function[::-1]
    # print(blur_function)
    smoothed_blur = np.zeros(shape=blur_function.shape)
    window_size = 5  # 2
    for i in range(len(smoothed_blur)):
        if i < window_size:
            for j in range(i):
                smoothed_blur[i] = smoothed_blur[i] + blur_function[j]
            smoothed_blur[i] = smoothed_blur[i] / (i + 1)
        else:
            for j in range(window_size):
                smoothed_blur[i] = smoothed_blur[i] + blur_function[i - j]
            smoothed_blur[i] = smoothed_blur[i] / window_size

    plt.plot(smoothed_blur)
    plt.show()
    actual_blur = smoothed_blur
    actual_blur = actual_blur[1:intensity + 1]
    actual_blur = actual_blur / max(actual_blur)
    rotated_Image = rotation(image, min_ind / 36 * (2 * np.pi))
    actual_blur = np.ones(intensity)

    length_temp_col = len(rotated_Image)
    length_temp_row = len(rotated_Image[0])
    rotated_Image = np.pad(rotated_Image, ((0, length_temp_col), (0, length_temp_row)), 'constant',
                           constant_values=(0, 0))
    plt.imshow(rotated_Image)
    plt.show()

    padding = np.zeros(shape=rotated_Image.shape)
    # padding2 = np.zeros(int(len(rotated_Image[0]))//2)
    # print(padding[0,0:len(actual_blur)])
    padding[0, 0:len(actual_blur)] = padding[0, 0:len(actual_blur)] + actual_blur[0:len(actual_blur)]
    # actual_blur = np.fft.fftshift(actual_blur)
    # plt.show()
    print(padding.shape)
    print(rotated_Image.shape)
    result_image = MMSE_filter(rotated_Image, padding, 10)

    result_image_list = hist_equ_11812417(normalization(result_image[0:length_temp_col, 0:length_temp_row]))
    plt.imshow(result_image_list[0])
    plt.show()
    plt.imshow(result_image_list[0], cmap='gray')
    plt.show()

    # print(blur_function)

    # plt.plot(np.abs(shifted_MTF))
    # plt.show()

    # output_image = rotation(image, np.pi/2)
    # print(output_image)
    # list_test = findDirection(image)
    # print(np.argmin(list_test))
    # print(list_test)
    # output_image = derivative(image,0)
    # plt.imshow(output_image, cmap='gray')
    # plt.show()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
