import numpy as np
import math
import cv2 as cv
from numpy import array
from PIL import Image as im
np.set_printoptions(threshold=np.inf)




def image_to_array(image):

    gray =  cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    array2D = np.array(gray)
    image_array = array2D.flatten()
    np.save('image_coeffs', image_array)
    return image_array    


def printKthBit(n, k):
    return((n & (1 << (k - 1))) >> (k - 1))


def Significance(coeffs,start_index, lambda_, T):
    # print(f'Significance Call')
    max_abs = np.max(np.abs(coeffs[start_index: start_index + lambda_]))
    if max_abs < T:
        return 0
    elif T <= max_abs < 2 * T:
        return 1
    else:
        return None


def Significance_PScan(index, T):
    # print(f'Significance Call')
    if abs(index) < T:
        return 0
    elif T <= abs(index) < 2 * T:
        return 1
    else:
        return None


def Process_I(coeffs, start_index, lambda_, T):
    print(f'Process I call')
    output = Significance(coeffs, start_index, start_index  + lambda_, T)
    if output == 0:
        start_index = lambda_
    return start_index


def Pscan(coeffs, start_index, T, b):
    global output
    print(f'PScan Call')
    for j in range(4):
        significance = Significance_PScan(coeffs[start_index + j], T)
        # print(f'Significance : {significance}')
        if significance == 1 or significance == 0:
            output = output + str(significance)
            print(f'1. Significance {output}')
            if len(output) == 8:
                print('inserting into file')
                with open("output.txt", 'a') as file:
                    # file.write('-')
                    file.write(output)
                output = ''

        if significance == 1:

            if coeffs[start_index + j] < 0:
                sign_bit = 1
            else:
                sign_bit = 0
            output = output + str(sign_bit)
            print(f'2. Sign Bit {output}')
            if len(output) == 8:
                print('inserting into file')
                with open("output.txt", 'a') as file:
                    # file.write('-')
                    file.write(output)
                output = ''
            # print("MSB:", msb)  # Output bth MSB of block
        elif significance == None:
            msb = printKthBit(coeffs[start_index + j], b + 1)
            output = output + str(msb)
            print(f'3. MSB {output}')
            if len(output) == 8:
                print('inserting into file')
                with open("output.txt", 'a') as file:
                    # file.write('-')
                    file.write(output)
                output = ''

    return start_index + 4


def EvalSL(start_index, lambda_):
        print(f'Process Eval SL call')
        if start_index & 3 * lambda_ == 0:
            lambda_ = lambda_ * 4
        return lambda_
    

def process_S(coeffs, start_index, lambda_, T, b):

    global output
    print(f"In process S, Start Index: {start_index}, Size: {lambda_}, Threshold: {T}")
    # print("Output Array:", output)
    significance = Significance(coeffs, start_index, lambda_, T)
    # print(f'Signigicance: {significance}')
    if significance == 0:
        print(f'significance start index {start_index}')
        for i in range(start_index, start_index + lambda_):
            output = output + str(significance)
            print(f'4. Significance {output}')
            if len(output) == 8:
                print('inserting into file')
                with open("output.txt", 'a') as file:
                    # file.write('--')
                    file.write(output)
                output = ''
        # print(f'hellooooo {output}')
        start_index = start_index + lambda_
        print(f'start index return if significance == 0 : {start_index}')

    else:
        if lambda_ > 4:
            lambda_ = lambda_//4
            # print(f'lambda gets divided: now -> {lambda_}')
            return start_index, lambda_
            # process_S(coeffs, start_index, lambda_, output, T)
        else:
            print(f'Start Index before Pscan: {start_index}')
            start_index = Pscan(coeffs, start_index, T, b)
            print(f'Start index after Pscan: {start_index}')

    lambda_ = EvalSL(start_index, lambda_)
    print(f'lambda returned from eval SL: {lambda_}')
    print(f'updated block : {output}')
    
    return start_index, lambda_


output = ''

def main():
    lower_bound = 0  # Adjust the lower bound as needed
    upper_bound = 255 # Adjust the upper bound as needed
    image = cv.imread('cameraman.png')
    coeffs = image_to_array(image)

    with open('image_coeffs.txt', 'w') as file:
        file.write(str(coeffs))

    Npix = len(coeffs)
    # print(Npix)

    L = 4
    lambda_root = Npix//4**L

    print(f'Coeffs: {coeffs}')
     
    print("Npix:", Npix)
    
    # Initialize b and threshold
    initial_b = math.floor(math.log2(np.max(np.abs(coeffs))))
    b = initial_b
    # print(f'bbbb {b}')  
    # output = np.empty(Npix, dtype="int")
    global output
    # output.fill(-1)
    byte = ''
    
    open('output.txt', 'w').close()

    #print(output)
    
    while b >= 0:
        start_index = 0
        lambda_ = lambda_root
        T = 2**b
        
        while start_index < Npix:
            print(f'New Iteration')
            start_index, lambda_ = process_S(coeffs, start_index, lambda_, T, b)
            if start_index == lambda_ and start_index >= lambda_root:
                start_index = Process_I(coeffs, start_index, lambda_, T)
                print(f'start index returned from Process I: {start_index}')
            print(f'Next Starting Index: {start_index}, Next Lambda: {lambda_}\n\n\n')
        
        b = b - 1

    with open("output.txt", 'a') as file:
        # file.write('--')
        file.write(output)
    output = ''


if __name__ == '__main__':
    main()