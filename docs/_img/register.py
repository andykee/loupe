import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import loupe

IMG_FILENAME = ''
SHIFT = (-24.8, 36.2)
OVERSAMPLE = 10

ref = np.array(Image.open(IMG_FILENAME).convert('L'))

arr = loupe.shift(ref, SHIFT)

F = np.fft.fft2(arr)
G = np.fft.fft2(ref)
xcorr = np.fft.fftshift(np.fft.ifft2(G*np.conj(F)))

maxima = np.unravel_index(np.argmax(np.abs(xcorr)), xcorr.shape)
peak = xcorr[maxima]

# compute shifts
center = np.array([np.fix(x/2) for x in arr.shape])
shift = maxima - center

npix_dft = np.ceil(OVERSAMPLE*1.5)
dft_shift = np.fix(npix_dft/2)
rs = dft_shift - shift[0] * OVERSAMPLE
cs = dft_shift - shift[1] * OVERSAMPLE

# Compute DFT
X = np.arange(arr.shape[1]) - np.floor(arr.shape[1]/2)
Y = np.arange(arr.shape[0]) - np.floor(arr.shape[0]/2)
U = np.arange(npix_dft) - cs
V = np.arange(npix_dft) - rs
E1 = np.exp(-2*np.pi*1j/(arr.shape[0]*OVERSAMPLE)*np.outer(V,Y))
E2 = np.exp(-2*np.pi*1j/(arr.shape[1]*OVERSAMPLE)*np.outer(X,U))
xcorr2 = np.dot(np.dot(E1,np.conj(np.fft.ifftshift(G*np.conj(F)))),E2)


fig = plt.figure(figsize=(8, 3))

ax1 = plt.subplot(1, 4, 1)
ax2 = plt.subplot(1, 4, 2, sharex=ax1, sharey=ax1)
ax3 = plt.subplot(1, 4, 3)
ax4 = plt.subplot(1, 4, 4)

ax1.imshow(ref, cmap='gray')
ax1.set_axis_off()
ax1.set_title('Reference image')

ax2.imshow(arr, cmap='gray')
ax2.set_axis_off()
ax2.set_title('Shifted image')

ax3.imshow(xcorr.real)
ax3.set_axis_off()
ax3.set_title('Cross-correlation')

ax4.imshow(xcorr2.real)
ax4.set_axis_off()
ax4.set_title('Oversampled XC')

plt.savefig('../_static/img/register.png', dpi=300, bbox_inches='tight', transparent=True)