import numpy as np
from math import sqrt
from PIL import Image as im 
import cv2 as cv
np.set_printoptions(threshold=np.inf)

array = []

def inverse_ProcessI(input_array, output_array, Npix, output_index, input_index, b):
    # print('process I called')
    if input_array[input_index] == 0:
        for lambda_i in range(output_index, Npix):
            if output_array[lambda_i] != 0:
                output_array[lambda_i] = output_array[lambda_i] - (2**b)/2
        # print(f'{output_index}')
        output_index = Npix
        # print(f'{output_index}')
    input_index = input_index + 1
    return output_index, input_index


def inverse_EvalSL(output_index, lambda_):
    while output_index & 3 * lambda_ == 0: 
        # print(f'output index: {output_index}, lambda_: {lambda_}')
        lambda_ = lambda_ * 4
        # print(f'new lambda: {lambda_}')
    return lambda_


def inverse_PScan(input_array, input_index, output_array, output_index, b):
    for lambda_i in range(output_index, output_index + 4):
                if input_array[input_index] == 0:
                    if output_array[lambda_i] != 0:
                        output_array[lambda_i] = output_array[lambda_i] - (2**b)/2
                    # if output_array[lambda_i] < 0:
                    #     output_array[lambda_i] = output_array[lambda_i] + (2**b)/2
                if input_array[input_index] == 1:
                    if output_array[lambda_i] == 0:
                        if input_array[input_index + 1] == 0:
                            output_array[lambda_i] = (2**b * float(1.5))
                        else:
                            output_array[lambda_i] = -(2**b * float(1.5))
                        input_index = input_index + 1
                    else:
                        # if output_array[lambda_i] > 0:
                            output_array[lambda_i] = output_array[lambda_i] + (2**b)/2
                        # if output_array[lambda_i] < 0:
                            # output_array[lambda_i] = output_array[lambda_i] - (2**b)/2
                input_index = input_index + 1

    return output_index + 4, input_index

def inverse_ProcessS(input_array, input_index, output_array, output_index, lambda_, b):
    if input_array[input_index] == 0:
        # print('Insignificant set')
        for lambda_i in range(output_index, output_index + lambda_):
            if output_array[lambda_i] != 0:
                output_array[lambda_i] = output_array[lambda_i] - (2**b)/2 
            # if output_array[lambda_i] < 0:
            #     output_array[lambda_i] = output_array[lambda_i] + (2**b)/2  
        output_index = output_index + lambda_
        input_index = input_index + 1
        # print(f'new output index: {output_index}')
    else:
        # print('Significant set')
        if lambda_ > 4:
            # print(f'lambda graeater than 4: {lambda_}')
            lambda_ = lambda_ // 4
            # print(f'new lambda: {lambda_}')
            input_index = input_index + 1
            return output_index, lambda_, input_index
        else:
            # print(f'Lambda is 4:')
            input_index = input_index + 1
            output_index, input_index = inverse_PScan(input_array, input_index, output_array, output_index, b)
    
    lambda_ =  inverse_EvalSL(output_index, lambda_)

    return output_index, lambda_, input_index


def main():
    with open('cameraman.bin', 'rb') as file:
        data = file.read()

    input_string = ''

    for nums in data:
        bin_value = bin(nums)
        input_string = input_string + str(bin_value[2:].zfill(8))
    # print(input_string)
    input_array = []

    for i in input_string:
        i = int(i)
        input_array.append(i)
    print(f'size of input array: {len(input_array)}')

    b = 10
    Npix = 256*256
    output_array = np.empty(Npix, dtype = 'int')
    output_array.fill(0)
    L = 3

    input_index  = 0
    lambda_root = Npix//4**L
    # print(output)
    while b >= 0:
        if (input_index >= len(input_array)):
            break
        print(f'For threshold: {2**b}')
        # print(f'Input index starts at: {input_index}')
        
        lambda_ = lambda_root
        output_index = 0
        # print(f'Output index at start: {output_index}')
        while output_index < Npix:

            
            output_index, lambda_, input_index = inverse_ProcessS(input_array, input_index, output_array, output_index, lambda_, b)
            # print(f'Returned from ProcessS\nOutput Index: {output_index}, Input Index: {input_index}, Lambda_: {lambda_}')
            if output_index == lambda_ and output_index >= lambda_root  and output_index < Npix:
                output_index, input_index = inverse_ProcessI(input_array, output_array, Npix, output_index, input_index, b)  
                # print(f'Returned from ProcessI\nOutput Index: {output_index}, Input Index: {input_index}, Lambda_: {lambda_}')

        b = b - 1
        print(f'next input index: {input_index}')
    
    np.save('result_decoder', output_array)
    output_array = np.array(output_array)
    print(output_array)

    a = int(sqrt(len(output_array)))
    image_matrix = np.reshape(output_array, (a, a))

    from matplotlib import pyplot as plt

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
            

