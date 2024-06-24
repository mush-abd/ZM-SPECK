import numpy as np
from math import sqrt
from PIL import Image as im 
import cv2 as cv
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.inf)


def main():

    with open('binary_encoded.bin', 'rb') as file:
        data = file.read()

    input_string = ''

    for nums in data:
        bin_value = bin(nums)
        # print(bin_value[2:])
        input_string = input_string + str(bin_value[2:].zfill(8))

    
    input_array = []

    for i in input_string:
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
    
    np.save('decoder_output', output)
    output_array = np.array(output)

    # print(output_array)
    a = int(sqrt(len(output_array)))
    image_matrix = np.reshape(output_array, (a, a))
    image_matrix_normalized = cv.normalize(image_matrix, None, 0, 255, cv.NORM_MINMAX)
    # Converting to uint8 for proper image display
    image_matrix_uint8 = image_matrix_normalized.astype('uint8')

    image = im.fromarray(image_matrix_uint8, 'L')
    image.show()
    image.save('reconstructed.png')

    plt.imshow(image_matrix_uint8, cmap='gray', interpolation='nearest')
    plt.show()





if __name__ == '__main__':
    main()
            

