import numpy as np
from imageio import imread, imsave
import matplotlib.pyplot as plt
import sys
from scipy.signal import fftconvolve

bsplinelinear = np.array([[1 / 4, 1 / 2, 1 / 4],
                          [2**(1 / 2) / 4, 0, -2**(1 / 2) / 4],
                          [-1 / 4, 1 / 2, -1 / 4]])
bsplinecubic = np.array([[1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16],
                         [1 / 16, -1 / 4, 3 / 8, -1 / 4, 1 / 16],
                         [-1 / 8, 1 / 4, 0, -1 / 4, 1 / 8],
                         [6**(1 / 2) / 16, 0, -6**(1 / 2) /
                          8, 0, 6**(1 / 2) / 16],
                         [-1 / 8, -1 / 4, 0, 1 / 4, 1 / 8]])
haar = np.array([[1 / 2, 1 / 2],
                 [1 / 2, -1 / 2]])
db2 = np.array([[-0.12940952255092145, 0.22414386804185735,
                 0.836516303737469, 0.48296291314469025],
                [-0.48296291314469025, 0.836516303737469,
                 -0.22414386804185735, -0.12940952255092145]] / np.sqrt(2))
db3 = np.array([[0.035226291882100656, -0.08544127388224149, -0.13501102001039084,
                 0.4598775021193313, 0.8068915093133388, 0.3326705529509569],
                [-0.3326705529509569, 0.8068915093133388, -0.4598775021193313,
                 -0.13501102001039084, 0.08544127388224149, 0.035226291882100656]] / np.sqrt(2))

def _downsample(matrix, offset):
    return matrix[0 + offset::2, 0 + offset::2]

def _upsample(matrix, size, offset):
    temp = np.zeros(size)
    temp[0 + offset::2, 0 + offset::2] = matrix
    return temp

# 存储小波系数的结构
class WaveletCoeffs:
    # 给定图像大小，计算系数矩阵的大小
    def __getsizes(self, itr):
        if itr == self.levels + 1:
            return
        ver, horiz = self.sizes[-1]
        # 与掩模卷积后得到尺寸
        ver, horiz = ver + self.masklen - 1, horiz + self.masklen - 1
        ver1, horiz1 = ver // 2, horiz // 2
        # 对于偶数长度掩码，我们需要放弃偶数索引行/列
        parity = (self.masklen + 1) % 2
        if parity == 0 and ver % 2 == 1:
        # 如果总数是奇数并且从零开始，我们需要一个额外的行/列
            ver1 += 1
        if parity == 0 and horiz % 2 == 1:
            horiz1 += 1
        self.sizes.append((ver1, horiz1))
        self.__getsizes(itr + 1)

    def __init__(self, masks, levels, size):
        self.masks = np.array(masks, dtype=np.float64)
        self.nummasks, self.masklen = self.masks.shape
        self.levels = levels
        # 每个阶段系数数组的大小
        self.sizes = [size, ]
        self.__getsizes(1)
        self.coeffs = np.empty(levels + 1, dtype=object)
        for i in range(levels + 1):
            if i == 0:
                self.coeffs[i] = np.empty((1, 1) + size)
            else:
                self.coeffs[i] = np.empty(
                    (self.nummasks, self.nummasks) + self.sizes[i])

    def __add__(self, other):
        result = WaveletCoeffs(self.masks, self.levels, self.sizes[0])
        result.coeffs = self.coeffs + other.coeffs
        return result

    # 重载小波系数的“-”运算符（和 + 类似）
    def __sub__(self, other):
        result = WaveletCoeffs(self.masks, self.levels, self.sizes[0])
        result.coeffs = self.coeffs - other.coeffs
        return result

    # 重载小波系数的“*”运算符（和 + 类似）
    def __mul__(self, other):
        result = WaveletCoeffs(self.masks, self.levels, self.sizes[0])
        result.coeffs = self.coeffs * other
        return result

    def __truediv__(self, other):
        result = WaveletCoeffs(self.masks, self.levels, self.sizes[0])
        result.coeffs = self.coeffs / other
        return result

    def __getitem__(self, key):
        if isinstance(key, int) or isinstance(key, slice) or isinstance(key, list):
            return self.coeffs[key]
        elif isinstance(key, tuple):
            if len(key) == 1:
                return self.coeffs[key]
            elif len(key) > 1:
                return self.coeffs[key[0]][key[1:]]
        else:
            raise IndexError('Index must be an integer, a slice, a list of integers or a tuple of indices')

    def __setitem__(self, key, value):
        if isinstance(key, int) or isinstance(key, slice) or isinstance(key, list):
            self.coeffs[key] = value
        elif isinstance(key, tuple):
            if len(key) == 1:
                self.coeffs[key] = value
            elif len(key) > 1:
                self.coeffs[key[0]][key[1:]] = value
        else:
            raise IndexError('Index must be an integer, a slice, a list of integers or a tuple of indices')

    # 二维矩阵的离散小波逆变换
    def invdwt2(self):
        temp = self[self.levels, 0, 0]
        for cur_level in range(self.levels, 0, -1):
            nummasks = self.nummasks
            masks = self.masks
            masklen = self.masklen
            offset = (masklen + 1) % 2
            # 上采样计算正确的大小
            final_x, final_y = self.sizes[cur_level]
            prev_x, prev_y = self.sizes[cur_level - 1]
            final_x, final_y = 2 * final_x, 2 * final_y
            # 下采样前的卷积矩阵有奇数行 (X+masklen-1)，舍弃了偶数行
            if prev_x % 2 == 0 and masklen % 2 == 0:
                final_x += 1
            # 矩阵有奇数行，舍弃了奇数行
            elif prev_x % 2 == 1 and masklen % 2 == 1:
                final_x -= 1
            if prev_y % 2 == 0 and masklen % 2 == 0:
                final_y += 1
            elif prev_y % 2 == 1 and masklen % 2 == 1:
                final_y -= 1
            for mask1 in range(nummasks):
                for mask2 in range(nummasks):
                    if mask1 == mask2 == 0:
                        temp = 2 * fftconvolve(
                            masks[mask1, :, None] * masks[mask2],
                            _upsample(temp, (final_x, final_y), offset))[
                                masklen - 1:-(masklen - 1), (masklen - 1):-(masklen - 1)]
                    else:
                        temp += 2 * fftconvolve(
                            masks[mask1, :, None] * masks[mask2],
                            _upsample(self[cur_level, mask1, mask2], (final_x, final_y), offset))[
                                masklen - 1:-(masklen - 1), (masklen - 1):-(masklen - 1)]
        return temp

# 二维矩阵的离散小波变换
def dwt2(img, masks, levels):
    img = np.array(img)
    masks = np.array(masks)
    # mask 数量和每个 mask 的长度
    nummasks, masklen = masks.shape
    offset = (masklen + 1) % 2
    imgdwt = WaveletCoeffs(masks, levels, img.shape)
    imgdwt[0, 0, 0] = img
    for cur_level in range(1, levels + 1):
        for mask1 in range(nummasks):
            for mask2 in range(nummasks):
                imgdwt[cur_level, mask1, mask2] = _downsample(
                    2 * fftconvolve(masks[mask1, -1::-1, None] * masks[mask2, -1::-1],
                                    imgdwt[cur_level - 1, 0, 0]), offset)

    return imgdwt

def prox(wcoeffs, thresh): 
    if not isinstance(wcoeffs, WaveletCoeffs):
        raise TypeError('First argument must be of type waveletCoeffs')
    temp = WaveletCoeffs(wcoeffs.masks, wcoeffs.levels, wcoeffs.sizes[0])
    for lev in range(1, wcoeffs.levels + 1):
        thresh_mask = (np.abs(wcoeffs[lev]) >
                       thresh[lev - 1]).astype(np.float64)
        temp[lev] = (wcoeffs[lev] - thresh_mask *
                     np.sign(wcoeffs[lev]) * thresh[lev - 1]) * thresh_mask
    return temp

def apgd(f, A, thresh=[0.1, 0.07, 0.04, 0.01], masks=bsplinelinear, levels=4, iters=20, verbose=True, showiters=False):
    # APGD 算法
    if verbose:
        sys.stdout.write("Running the APGD algorithm:\n")
    x_k = dwt2(f, masks, levels)
    x_km1 = x_k 
    t_k = 1
    t_km1 = 0
    n = levels
    h = np.array(masks)
    L = 1.0
    thresh = np.array(thresh)
    for k in range(1, iters + 1):
        if verbose:
            sys.stdout.write('\rIteration ' + str(k) + ' of ' + str(iters))
            sys.stdout.flush()
        y_k = x_k - ((x_k - x_km1) * (np.ones(n + 1) * (t_km1 - 1) / t_k))
        g_k = y_k - (dwt2(A * (A * y_k.invdwt2() - f), h, n)
                     * np.ones(n + 1) / L)
        x_kp1 = prox(g_k, thresh / L)
        t_kp1 = (1 + np.sqrt(1 + 4 * (t_k**2))) / 2
        x_km1 = x_k
        x_k = x_kp1
        t_km1 = t_k
        t_k = t_kp1
        if showiters:
            plt.clf()
            disp = x_k.invdwt2()
            disp[disp < 0] *= 0
            plt.imshow(disp, cmap=plt.cm.gray)
    if verbose:
        sys.stdout.write('\nDone\n')
    return x_k.invdwt2()

#对于已经有噪声的图片利用小波变换去噪
img = imread('input/2_input2.png')/255
A = 1-(img==1.0).astype(np.float64)
img*=A
img1=apgd(img, A)
imsave('output/2_output2.png', img1)

#加载图片
img = imread('input/2_input.jpg')/255


#增加噪声
noise_scale = 0.1
img_noisy = img + noise_scale*np.random.normal(size=img.shape)
imsave('output/2_output_noise.jpg', img_noisy)

# 利用小波变换去噪
img1 = apgd(img_noisy, np.ones(img.shape))
imsave('output/2_output_noise_corrected.jpg', img1)
