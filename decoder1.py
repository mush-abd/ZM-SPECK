import numpy as np
from math import sqrt
from PIL import Image as im 
import cv2 as cv
np.set_printoptions(threshold=np.inf)


def main():
    with open('fin_img.bin', 'rb') as file:
        data = file.read()

    input_string = ''

    for nums in data:
        bin_value = bin(nums)
        input_string = input_string + str(bin_value[2:].zfill(8))

    input_array = []

    for i in input_string:
        i = int(i)
        input_array.append(i)

    print(f'size of input array: {len(input_array)}')
    # print(len(input_array))

    b = 7
    Npix = 64*64
    output = np.empty(Npix, dtype = 'int')
    output.fill(0)
    L = 3

    j  = 0
    lambda_root = Npix//4**L
    # print(output)
    while b >= 0:
        if (j >= len(input_array)):
            break
        
        print(f'for threshold: {2**b}')
        print(f'Input index starts at: {j}')
        lambda_ = lambda_root
        # print(f'lambda_root: {lambda_}')
        output_index = 0
        while output_index < Npix:
            print(f'Output Index: {output_index}, Lambda_: {lambda_}')
            if lambda_ == 4:
                print('for lambda 4')
                if input_array[j] == 0:
                    print('for input_array = 0')
                    for lambda_i in range(output_index, output_index + 4):
                            output[lambda_i] = output[lambda_i] - (2**b)/2
                    j = j + 1
                    output_index = output_index + 4  
                elif input_array[j] == 1:
                    print(f'for input_array = 1')
                    j = j + 1
                    for lambda_i in range(output_index, output_index + 4):                        
                        if output[lambda_i] == 0:
                            if input_array[j] == 0:
                                if input_array[j + 1] == 0:
                                    output[lambda_i] = (2**b * float(1.5))
                                else:
                                    output[lambda_i] = -(2**b * float(1.5))
                                j = j + 1
                        elif output[lambda_i] != 0:
                            if input_array[j] == 0:
                                output[lambda_i] = output[lambda_i] - (2**b)/2
                            else:
                                output[lambda_i] = output[lambda_i] + (2**b)/2
                        j = j + 1
                    output_index = output_index + 4
                if output_index & 3 * lambda_ == 0:
                    lambda_ = lambda_ * 4
                # output_index += lambda_
                # output_index = output_index + lambda_ 
            else:
                print(f'lambda is greater than 4')
                if output_index == lambda_root and output_index >= lambda_root:
                    print('process I called')
                    if input_array[j] == 0:
                        for lambda_i in range(output_index, Npix):
                            if output[lambda_i] != 0:
                                output[lambda_i] = output[lambda_i] - (2**b)/2
                        print(f'{output_index}')
                        output_index = Npix
                        print(f'{output_index}')
                    else:
                        lambda_ = lambda_//4
                    j = j + 1
                elif input_array[j] == 1:
                    lambda_ = lambda_ //4
                    print(f'lambda becomes: {lambda_}')
                    j = j + 1
                elif  input_array[j] == 0:
                    print(f'Lambda not devided')
                    for lambda_i in range(output_index, output_index + lambda_):
                        if output[lambda_i] != 0:
                                output[lambda_i] = output[lambda_i] - (2**b)/2
                    j = j + 1
                    output_index = output_index + lambda_
                    if output_index & 3 * lambda_ == 0:
                        print(f'lambda increased')
                        lambda_ = lambda_ * 4
                # output_index += lambda_
                # output_index = output_index + lambda_ 
            # if output_index & 3 * lambda_ == 0:
            #     lambda_ = lambda_ * 4
            #     output_index += lambda_
            #     output_index = output_index + lambda_   

        b = b - 1
        print(f'next input index: {j}')
    
    np.save('result_decoder', output)
    output_array = np.array(output)
    print(output_array)

    a = int(sqrt(len(output_array)))
    image_matrix = np.reshape(output_array, (a, a))

    # # print(image.shape)
    # print(image_matrix)
    # print(a, image_matrix.shape)

    # image = im.fromarray(image_matrix, 'L')
    # image.show()
    # cv.waitKey(0)
    # image.save('reconstructed.png')
    from matplotlib import pyplot as plt
    # plt.imshow(image_matrix, interpolation='nearest')
    # plt.show()
    image_matrix_normalized = cv.normalize(image_matrix, None, 0, 255, cv.NORM_MINMAX)

    # Converting to uint8 for proper image display
    image_matrix_uint8 = image_matrix_normalized.astype('uint8')

    image = im.fromarray(image_matrix_uint8)
    image.show()
    image.save('reconstructed.png')

    # plt.imshow(image_matrix_uint8, cmap='gray', interpolation='nearest')
    # plt.show()





if __name__ == '__main__':
    main()
            

