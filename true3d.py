import numpy as np
import nrrd
import poisson3d
import poissonblending
import PIL.Image

dx = 100; N = 400; d = 20

#prepare(r'Images/Carb.nhdr', r'Images/Carb2.nhdr')
img_source, image_header = nrrd.read(r'source.nhdr')
img_target, image_header = nrrd.read(r'target.nhdr')
offset = int(img_target.shape[0]/2 - 3*d/4)

sls = [	(slice(None), 0, slice(None)),
		(slice(None), -1, slice(None)),
		(slice(None), slice(None), 0),
		(slice(None), slice(None), -1)]

for sl in sls:
	img_boundary = img_target[sl]
	img_boundary_source = img_source[sl]
	img_mask = np.zeros(img_boundary_source.shape)
	img_mask[ 1:-1 , 1:-1] = 255
	boundary_res = poissonblending.blend(img_boundary, img_boundary_source, img_mask, offset = (offset, 0))
	#boundary_res[boundary_res>=0.5] = 1
	#boundary_res[boundary_res<0.5] = 0
	img_ret = PIL.Image.fromarray(np.uint8(boundary_res))
	img_ret.save('./testimages/bound' + str(sls.index(sl)) + '.png')
	img_target[sl] = boundary_res

img_mask = np.zeros(img_source.shape)
img_mask[1:-1,1:-1,1:-1] = 255
data = poisson3d.blend(img_target, img_source, img_mask, offset = (offset,0,0))
options = {'encoding' : 'raw'}
nrrd.write(r'result.nhdr', data, options = options)
