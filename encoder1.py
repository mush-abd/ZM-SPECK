import numpy as np
import math
import cv2 as cv
from numpy import array
from PIL import Image as im
np.set_printoptions(threshold=np.inf)
import struct

counter = 0
def write_to_file(bin):
    # global arr
    global counter
    
    while len(bin) < 8:
        bin = bin + '0'
    # counter = counter + 8
    bin_int = int(bin, 2)
    counter = counter + 8
    byte_num = struct.pack('B', bin_int)
    with open('fin_img.bin', 'ab') as file:
        file.write(byte_num)
    bin = ''
    return bin

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
    # print(f'Significance PScan Call')
    if abs(index) < T:
        return 0
    elif T <= abs(index) < 2 * T:
        return 1
    else:
        return None

def Significance_I(coeffs, start_index, Npix, T):
    # print(f'Significance I call')
    # print(coeffs[start_index: Npix - 1])
    max_abs = np.max(np.abs(coeffs[start_index: Npix - 1]))
    if max_abs < T:
        return 0
    elif T <= max_abs < 2 * T:
        return 1
    else:
        return None

def Process_I(coeffs, start_index, Npix, T):
    # print(f'Process_I call')
    global output
    # if coeffs[start_index:Npix - 1]  is not []:
    significance = Significance_I(coeffs, start_index, Npix - 1, T)
    if significance == 0:
        output = output + str(significance)
        if len(output) == 8:
            output = write_to_file(output)
        return Npix


def Pscan(coeffs, start_index, T, b):
    # print(f'PSCan Call: ')
    global output
    for j in range(4):
        significance = Significance_PScan(coeffs[start_index + j], T)
        if significance == 1 or significance == 0:
            output = output + str(significance)
            if len(output) == 8:
                output = write_to_file(output)

        if significance == 1:
            if coeffs[start_index + j] < 0:
                sign_bit = 1
            else:
                sign_bit = 0
            output = output + str(sign_bit)
            if len(output) == 8:
                output = write_to_file(output)

        elif significance == None:
            msb = printKthBit(coeffs[start_index + j], b + 1)
            output = output + str(msb)
            if len(output) == 8:
                output = write_to_file(output)
    return start_index + 4


def EvalSL(start_index, lambda_):
        # print(f'EvalSL call')
        if start_index & 3 * lambda_ == 0:
            lambda_ = lambda_ * 4
        # print(f'Lambda return from EvalSL: {lambda_}')
        return lambda_

    

def process_S(coeffs, start_index, lambda_, T, b):
    # print(f'Process_S call')
    global output
    significance = Significance(coeffs, start_index, lambda_, T)
    if significance is not None:
        output = output + str(significance)
        if len(output) == 8:
            output = write_to_file(output)
    if significance == 0:
        start_index = start_index + lambda_
    else:
        if lambda_ > 4:
            lambda_ = lambda_//4
            return start_index, lambda_
        else:
            start_index = Pscan(coeffs, start_index, T, b)

    lambda_ = EvalSL(start_index, lambda_)
    # print(f'Return from Process_S, Start Index : {start_index}, Lambda: {lambda_}')
    return start_index, lambda_


output = ''

def main():

    coeffs = np.load('array_coeffs.npy')
    print(coeffs.shape)
    Npix = len(coeffs)

    L = 3
    lambda_root = Npix//4**L

    # print(f'Coeffs: {coeffs}')
    print("Npix:", Npix)
    
    # Initialize b and threshold
    initial_b = math.floor(math.log2(np.max(np.abs(coeffs))))
    b = initial_b
    print(f'b: {b}')  
    global output
    
    open('fin_img.bin', 'wb').close()
    
    while b >= 0:
        print(f'Threshold: {2**b}')
        start_index = 0
        lambda_ = lambda_root
        T = 2**b
        
        while start_index < Npix:
            # print(f'Starting Index: {start_index}')
            start_index, lambda_ = process_S(coeffs, start_index, lambda_, T, b)
            if start_index == lambda_ and start_index >= lambda_root and start_index < Npix:
                # print(f'Process_I called from main')
                start_index = Process_I(coeffs, start_index, Npix, T)
                # print(f'start index returned from Process I: {start_index}')
            # print(f'Next Starting Index: {start_index}, Next Lambda: {lambda_}\n\n\n')
        
        b = b - 1

    output = write_to_file(output)
    print(f'Counter : {counter}')


if __name__ == '__main__':
    main()