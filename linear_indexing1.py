import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

l = [1/math.sqrt(2),1/math.sqrt(2)]
h = [-1/math.sqrt(2),1/math.sqrt(2)]

def convp(s, filter):
    result = np.convolve(s, filter)[1::2]
    return result

def frwt(s, l, h):
    cA = convp(s, l)
    cD = convp(s, h)

    return cA, cD

def frwt2(s, l, h):
    L = []
    H = []
    for row in s:
        cA, cD = frwt(row, l, h)
        L.append(cA)
        H.append(cD)
    return L, H

def frwt_2D(s, l, h):
    L, H = frwt2(s, l, h)

    LT = np.transpose(L)
    HT = np.transpose(H)

    LL, LH = frwt2(LT, l, h)
    HL, HH = frwt2(HT, l, h)

    return np.array(np.transpose(LL)) ,np.array(np.transpose(LH)) ,np.array(np.transpose(HL)) ,np.array(np.transpose(HH))
img = cv2.imread( "image_resized_2048.png", cv2.IMREAD_GRAYSCALE)
print(img.shape)
cA1 , cH1 , cV1 , cD1 = frwt_2D(img,l,h)

cA2 , cH2 , cV2 , cD2 = frwt_2D(cA1,l,h)

cA3 , cH3 , cV3 , cD3 = frwt_2D(cA2,l,h)

print(cA1.shape,cH1.shape,cV1.shape,cD1.shape)
print(cA2.shape,cH2.shape,cV2.shape,cD2.shape)
print(cA3.shape,cH3.shape,cV3.shape,cD3.shape)
q1_half_up = np.concatenate((cA3,cV3),axis=1)
q1_half_down = np.concatenate((cH3,cD3),axis=1)
q1_half = np.concatenate((q1_half_up,q1_half_down),axis=0)

q1_up = np.concatenate((q1_half, cV2),axis=1)
q1_down = np.concatenate((cH2, cD2),axis=1)
q1 = np.concatenate((q1_up, q1_down),axis=0)

up = np.concatenate((q1,cV1),axis=1)
down = np.concatenate((cH1,cD1),axis=1)
final_image = np.concatenate((up,down),axis=0)
# fin_img = final_image.flatten()

# print(fin_img.shape, final_image.shape, len(fin_img))

# np.save('fin_img', fin_img)
plt.imshow(final_image,cmap="gray")
plt.axis('off')
plt.show()
def linear_indexing_wavelet_coeffs(image):
    # Assuming the input image is the output of 3-level fractional wavelet filtering
    L = 3  # Number of wavelet decomposition levels

    M, N = image.shape

    # Calculate the number of subbands in the transformed image
    num_subbands = 3 * L + 1

    # Initialize an array to store the linearly indexed coefficients
    linear_coeffs = np.zeros(M * N * (1 + 1 // 4 + 1 // 4 + 1 // 2), dtype=image.dtype)

    # Linear indexing for LL-subband (k = 1)
    linear_coeffs[: M * N] = image.flatten()

    index = M * N

    # Linear indexing for LH-subband (k = 2) and HL-subband (k = 3)
    for _ in range(2):
        p = 1  # Resolution level for LH and HL subbands
        set_length = M * N // (2 ** p)
        linear_coeffs[index : index + set_length] = image.flatten()[M * N : M * N + set_length]
        index += set_length

    # Linear indexing for HH-subband (k = 4)
    k = 4
    p = 2  # Resolution level for HH subband
    set_length = M * N // (2 ** (p + k - 1))
    linear_coeffs[index : index + set_length] = image.flatten()[M * N + M * N // 4 : M * N + M * N // 4 + set_length]

    return linear_coeffs

# Example usage:
linear_coeffs = linear_indexing_wavelet_coeffs(final_image)
linear_coeffs.shape
arr_new = np.empty(len(linear_coeffs), dtype='int')
for i in range(len(arr_new)):
    arr_new[i] = math.trunc(linear_coeffs[i])
np.save('array_coeffs', arr_new)
# print(arr_new)
# plt.imshow(np.reshape(arr_new ,(1024, 1024)),cmap='gray')
# plt.axis('off')
# plt.show()
# cv2.imwrite("linear_coeffs.png", img)