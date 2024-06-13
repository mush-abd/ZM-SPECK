import numpy as np
from math import sqrt
from PIL import Image as im 
import cv2 as cv
np.set_printoptions(threshold=np.inf)


def main():
    input_file = open('output.txt', 'r')
    input = input_file.read()
    
    input_array = []

    for i in input:
        i = int(i)
        input_array.append(i)

    print(f'size of input array: {len(input_array)}')
    # print(len(input_array))

    b = 7
    Npix = 256 * 256
    output = np.empty(Npix, dtype = 'int')
    output.fill(0)


    j  = 0

    while b >= 0:
    
        print(f'for threshold: {2**b}')
        for i in range (Npix):
            if output[i] == 0:
                if input_array[j] == 1:
                    if input_array[j + 1] == 0:
                        output[i] = (2**b * float(1.5))
                    else:
                        output[i] = -(2**b * float(1.5))
                    j = j + 1
            elif output[i] != 0:
                if input_array[j] == 0:
                    output[i] = output[i] - (2**b)/2
                else:
                    output[i] = output[i] + (2**b)/2

            j = j + 1
        b = b - 1
        print(f'next input index: {j}')
    
    np.save('result_decoder_3', output)
    output_array = np.array(output)
    # print(output_array)
    # a = int(sqrt(len(output_array)))
    # image_matrix = np.reshape(output_array, (a, a))

    # # print(image.shape)
    # print(image_matrix)
    # print(a, image_matrix.shape)

    # image = im.fromarray(image_matrix, 'L')
    # image.show()
    # cv.waitKey(0)
    # image.save('reconstructed.png')
    # from matplotlib import pyplot as plt
    # plt.imshow(image_matrix, interpolation='nearest')
    # plt.show()




if __name__ == '__main__':
    main()
            

