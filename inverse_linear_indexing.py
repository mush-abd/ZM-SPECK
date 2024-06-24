import numpy as np
import cv2
import math


l = [1 / math.sqrt(2), 1 / math.sqrt(2)]
h = [-1 / math.sqrt(2), 1 / math.sqrt(2)]

def reverse_linear_indexing_wavelet_coeffs(linear_coeffs, image_shape):
    M, N = image_shape
    image_size = M * N

    # Reshape the linearly indexed coefficients to their original structure
    reshaped_coeffs = linear_coeffs[:image_size].reshape((M, N))

    LH_length = image_size // 4
    HL_length = image_size // 4
    HH_length = image_size // 2

    LH_coeffs = linear_coeffs[image_size:image_size + LH_length].reshape((M // 2, N // 2))
    HL_coeffs = linear_coeffs[image_size + LH_length:image_size + LH_length + HL_length].reshape((M // 2, N // 2))
    HH_coeffs = linear_coeffs[image_size + LH_length + HL_length:].reshape((M // 2, N // 2))

    return reshaped_coeffs, LH_coeffs, HL_coeffs, HH_coeffs

def inverse_frwt2(LL, LH, HL, HH, l, h):
    L = np.concatenate((LL, LH), axis=1)
    H = np.concatenate((HL, HH), axis=1)
    LT = np.transpose(L)
    HT = np.transpose(H)
    L_inv, H_inv = [], []
    for row in LT:
        c_inv = np.convolve(row, l[::-1], 'full')[:-1] + np.convolve(row, h[::-1], 'full')[:-1]
        L_inv.append(c_inv)
    for row in HT:
        c_inv = np.convolve(row, l[::-1], 'full')[:-1] + np.convolve(row, h[::-1], 'full')[:-1]
        H_inv.append(c_inv)
    L_inv = np.transpose(np.array(L_inv))
    H_inv = np.transpose(np.array(H_inv))
    return L_inv + H_inv

def inverse_frwt(LL, LH, HL, HH, l, h):
    LL_inv = inverse_frwt2(LL, LH, HL, HH, l, h)
    inversed_image = inverse_frwt2(LL_inv, [], [], [], l, h)
    return inversed_image

# Load linear coefficients
linear_coeffs = np.load('result_decoder_3.npy')
print(len(linear_coeffs))
# Define image shape (assuming it's a square image)
image_shape = (256, 256)

# Reverse linear indexing
reshaped_coeffs, LH_coeffs, HL_coeffs, HH_coeffs = reverse_linear_indexing_wavelet_coeffs(linear_coeffs, image_shape)

# Inverse wavelet transform
reconstructed_image = inverse_frwt(reshaped_coeffs, LH_coeffs, HL_coeffs, HH_coeffs, l, h)

# Display or save the reconstructed image
cv2.imwrite("reconstructed_image.png", reconstructed_image)
