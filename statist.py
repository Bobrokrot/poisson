import numpy as np
from scipy import ndimage
import PIL.Image
import matplotlib.pyplot as plt

winsize = 59
#row = 250
d = 50; n = 5
sufx = ['', '_mixed', '_average', '_none']
fsource = './testimages/test' + str(n) + '_source.png'
ftarget = './testimages/test' + str(n) + '_target.png'
fret = lambda mode: './testimages/test' + str(n) + '_ret' + sufx[mode] + '.png'
img_source = np.asarray(PIL.Image.open(fsource))
img_target = np.asarray(PIL.Image.open(ftarget))
img_ret = np.asarray(PIL.Image.open(fret(1)))
img_retn = np.asarray(PIL.Image.open(fret(-1)))

img_source.flags.writeable = True
img_target.flags.writeable = True
img_ret.flags.writeable = True
img_retn.flags.writeable = True

x = 450

img1 = np.zeros((500,500))
img1[:,:x] = img_target[:, :x]
img1[:,x:] = img_source
img2 = np.zeros((500,500))
img2 = img_target[:,x:]

MEAN1 = []; MEAN2 = []; MEAN3 = []; MEANR1 = []; MEANR2 = []; MEANR3 = []
VAR1 = []; VAR2 = []; VAR3 = []; VARR1 = []; VARR2 = []; VARR3 = []
for row in range(500):
	signal = img_target[row, :]
	sig1 = signal[:x]
	sig2 = signal[x:]
	M1 = np.mean(sig1)
	M2 = np.mean(sig2)
	M = np.mean(signal)
	V1 = np.var(sig1)
	V2 = np.var(sig2)
	V = np.var(signal)
	MEAN1.append(M1)
	MEAN2.append(M2)
	MEAN3.append(M)
	VAR1.append(V1)
	VAR2.append(V2)
	VAR3.append(V)
	
	signal_res = img_ret[row, :]
	signal_res1 = signal_res[:x]
	signal_res2 = signal_res[x:]
	MR1 = np.mean(signal_res1)
	MR2 = np.mean(signal_res2)
	MR = np.mean(signal_res)
	VR1 = np.var(signal_res1)
	VR2 = np.var(signal_res2)
	VR = np.var(signal_res)
	MEANR1.append(MR1)
	MEANR2.append(MR2)
	MEANR3.append(MR)
	VARR1.append(VR1)
	VARR2.append(VR2)
	VARR3.append(VR)

print np.mean(MEAN1), np.mean(MEAN2), np.mean(MEAN3)
print np.mean(MEANR1), np.mean(MEANR2), np.mean(MEANR3)
print np.var(MEAN1), np.var(MEAN2), np.var(MEAN3)
print np.var(MEANR1), np.var(MEANR2), np.var(MEANR3)
print np.mean(VAR1), np.mean(VAR2), np.mean(VAR3)
print np.mean(VARR1), np.mean(VARR2), np.mean(VARR3)

bins = np.linspace(0, 256, 129)
a1, kek = np.histogram(MEAN1, bins)
a2, kek = np.histogram(MEAN2, bins)
a3, kek = np.histogram(MEAN3, bins)
ar1, kek = np.histogram(MEANR1, bins)
ar2, kek = np.histogram(MEANR2, bins)
ar3, kek = np.histogram(MEANR3, bins)

b1, kek = np.histogram(VAR1, bins)
b2, kek = np.histogram(VAR2, bins)
b3, kek = np.histogram(VAR3, bins)
br1, kek = np.histogram(VARR1, bins)
br2, kek = np.histogram(VARR2, bins)
br3, kek = np.histogram(VARR3, bins)

c1, kek = np.histogram(img1, bins)
c2, kek = np.histogram(img2, bins)
c3, kek = np.histogram(img_ret, bins)

plt.plot(bins[:-1], 1.*c1/sum(c1), color = 'red', label = "left")
plt.plot(bins[:-1], 1.*c2/sum(c2), color = 'blue', label = "right")
plt.plot(bins[:-1], 1.*c3/sum(c3), color = 'green', label = "blended")
#plt.plot(bins[:-1], 1.*a1/sum(a1), color = 'red', label = "left")
#plt.plot(bins[:-1], 1.*a2/sum(a2), color = 'blue', label = "right")
#plt.plot(bins[:-1], a3, color = 'green', label = "both")
#plt.plot(bins[:-1], ar1, 'r-', label = "left blend")
#plt.plot(bins[:-1], ar2, 'b-', label = "right blend")
#plt.plot(bins[:-1], 1.*ar3/sum(ar3), 'g-', label = "both blend")
#plt.plot(bins[:-1], b1, color = 'red', label = "left")
#plt.plot(bins[:-1], b2, color = 'blue', label = "right")
plt.legend(loc='upper right')
plt.show()

"""
mean1 = ndimage.generic_filter(img1, np.mean, size=(winsize,winsize))
mean2 = ndimage.generic_filter(img2, np.mean, size=(winsize,winsize))
outMean = ndimage.generic_filter(img_target, np.mean, size=(winsize,winsize))
outMean2 = ndimage.generic_filter(img_ret, np.mean, size=(winsize,winsize))

Ymean = mean1[row, :]
plt.plot(range(0,500), Ymean, color = 'red', label = "Left image")
Ymean2 = mean2[row, :]
plt.plot(range(x, x + 500), Ymean2, color = 'green', label = "Right image")
#plt.show()



Ymean = outMean[row, :]
plt.plot(Ymean, color = 'magenta', label = "Both no blending")
Ymean2 = outMean2[row, :]
plt.plot(Ymean2, color = 'blue', label = "Both with blending")
plt.legend(loc='upper right')
plt.show()
"""