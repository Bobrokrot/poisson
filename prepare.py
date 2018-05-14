"""
d = 20
fn1 = r'Ber.nhdr'
fn2 = r'Ber2.nhdr'

img1, image_header = nrrd.read(fn1)
img2, image_header = nrrd.read(fn2)
imgs = img1[:-d,:, :]
#imgt = np.zeros((img1.shape[0] + img2.shape[0] - d/2, img1.shape[1], img1.shape[2]))
imgt = np.append(img1[:-d/2, :, :], img2, axis = 0)

options = {'encoding' : 'raw'}
nrrd.write(r'source2.nhdr', imgs, options = options)

options = {'encoding' : 'raw'}
nrrd.write(r'target2.nhdr', imgt, options = options)
"""
import numpy as np
import nrrd

dx = 100; N = 400; d = 20
def prepare(fn1, fn2):
	"""
	region1 = (slice(0, dx - d/2), slice(0, dx), slice(0, dx))
	region2 = (slice(0, dx), slice(0, dx), slice(N-dx, N))
	regionS = (slice(dx - d/2, dx), slice(0, dx), slice(0, dx))
	image_data, image_header = nrrd.read('Berea.nhdr')
	image1 = image_data[region1]
	image2 = image_data[region2]
	"""
	region1 = (slice(-d/2), slice(None), slice(None))
	regionS = (slice(-d, None), slice(None), slice(None))
	image1, image_header = nrrd.read(fn1)
	image2, image_header = nrrd.read(fn2)
	
	img_source = image1[regionS]
	image1 = image1[region1]
	img_target = np.append(image1, image2, axis = 0)
	options = {'encoding' : 'raw'}
	nrrd.write(r'target.nhdr', img_target, options = options)
	options = {'encoding' : 'raw'}
	nrrd.write(r'source.nhdr', img_source, options = options)
	
if __name__ == '__main__':
	prepare(r'Carb.nhdr', r'Carb2.nhdr')