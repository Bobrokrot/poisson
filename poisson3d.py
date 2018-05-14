# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse
import PIL.Image
import cv2
import pyamg
import time


def blend(img_target, img_source, img_mask, offset=(0,0,0)):
	# compute regions to be blended
	region_source = (
			max(-offset[0], 0),
			max(-offset[1], 0),
			max(-offset[2], 0),
			min(img_target.shape[0]-offset[0], img_source.shape[0]),
			min(img_target.shape[1]-offset[1], img_source.shape[1]),
			min(img_target.shape[2]-offset[2], img_source.shape[2]))
	region_target = (
			max(offset[0], 0),
			max(offset[1], 0),
			max(offset[2], 0),
			min(img_target.shape[0], img_source.shape[0]+offset[0]),
			min(img_target.shape[1], img_source.shape[1]+offset[1]),
			min(img_target.shape[2], img_source.shape[2]+offset[2]))
			
	region_size = (region_source[3]-region_source[0], region_source[4]-region_source[1], region_source[5]-region_source[2])
	# clip and normalize mask image
	img_mask = img_mask[region_source[0]:region_source[3], region_source[1]:region_source[4], region_source[2]:region_source[5]]
	#print img_mask.shape
	#img_mask = prepare_mask(img_mask)
	#img_mask[img_mask==0] = False
	#img_mask[img_mask!=False] = True

	# create coefficient matrix
	
	ind = lambda z,y,x: x+y*region_size[2]+z*region_size[1]*region_size[2]
	
	A = scipy.sparse.identity(np.prod(region_size), format='lil')
	for z in range(region_size[0]):
		for y in range(region_size[1]):
			for x in range(region_size[2]):
				if img_mask[z,y,x] != 0 :
					index = ind(z,y,x)
					Np = 0
					if index+1 < np.prod(region_size):
						A[index, index+1] = -1
						Np+=1
					if index-1 >= 0:
						A[index, index-1] = -1
						Np+=1
					if index+region_size[2] < np.prod(region_size):
						A[index, index+region_size[2]] = -1
						Np+=1
					if index-region_size[2] >= 0:
						A[index, index-region_size[2]] = -1
						Np+=1
					if index+region_size[1]*region_size[2] < np.prod(region_size):
						A[index, index+region_size[1]*region_size[2]] = -1
						Np+=1
					if index-region_size[1]*region_size[2] >= 0:
						A[index, index-region_size[1]*region_size[2]] = -1
						Np+=1
					A[index, index] = Np
	A = A.tocsr()
	
	# create poisson matrix for b
	P = pyamg.gallery.poisson(img_mask.shape)

	# for each layer (ex. RGB)
	N = 1 if len(img_target.shape) < 4 else img_target.shape[4]
	for num_layer in range(N):
		# get subimages
		if N == 1:
			t = img_target[region_target[0]:region_target[3], region_target[1]:region_target[4], region_target[2]:region_target[5]]
			s = img_source[region_source[0]:region_source[3], region_source[1]:region_source[4], region_source[2]:region_source[5]]
		else:	
			t = img_target[region_target[0]:region_target[3], region_target[1]:region_target[4], region_target[2]:region_target[5],num_layer]
			s = img_source[region_source[0]:region_source[3], region_source[1]:region_source[4], region_source[2]:region_source[5],num_layer]
		t = t.flatten()
		s = s.flatten()

		# create b
		#A = P.tocsr()
		b = P * s
		for z in range(region_size[0]):
			for y in range(region_size[1]):
				for x in range(region_size[2]):
					if img_mask[z,y,x] == 0:
						index = ind(z,y,x)
						b[index] = t[index]
		
					
		# solve Ax = b
		start = time.time()
		x = pyamg.solve(A,b,verb=False,tol=1e-10)
		print time.time() - start
		#print region_target
		#x = seidel(A,b, 1e-10)

		# assign x to target image
		x = np.reshape(x, region_size)
		x[x>255] = 255
		x[x<0] = 0
		x = np.array(x, img_target.dtype)
		#print x.shape
		if N == 1:
			img_target[region_target[0]:region_target[3], region_target[1]:region_target[4], region_target[2]:region_target[5]] = x
		else:
			img_target[region_target[0]:region_target[3], region_target[1]:region_target[4], region_target[2]:region_target[5],num_layer] = x
			

	return img_target

if __name__ == '__main__':
	from tvtk.api import tvtk
	import numpy as np
	from mayavi import mlab
	d = 50; dx = 20 
	#img_mask = np.asarray(PIL.Image.open('./testimages/test4_mask.png'))
	#img_mask.flags.writeable = True
	img_source = np.asarray(PIL.Image.open('./testimages/test5_source.png'))
	img_source.flags.writeable = True
	img_target = np.asarray(PIL.Image.open('./testimages/test5_target.png'))
	img_boundary = np.asarray(PIL.Image.open('./testimages/test5_ret.png'))
	#img_target = np.asarray(PIL.Image.open('./testimages/frontend-large.jpg'))
	img_target.flags.writeable = True
	img_target = np.array([img_target for _ in range(dx)])
	img_target[0] = img_boundary
	img_target[-1] = img_boundary
	#img_mask = np.array([img_mask for _ in range(dx)])
	#img_mask[0,:,:] = 0; img_mask[-1,:,:] = 0
	img_source = np.array([img_source for _ in range(dx)])
	img_mask = np.zeros(img_source.shape)
	img_mask[1:-1,1:-1,1:-1] = 255
	
	
	data = blend(img_target, img_source, img_mask, offset = (0,0,int(img_target.shape[2]/2 - 3*d/4)))[:dx/2,:,:]
	
	#data = img_target
	i = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0))
	i.point_data.scalars = data.ravel()
	i.point_data.scalars.name = 'scalars'
	i.dimensions = (data.shape[2], data.shape[1], data.shape[0])
	mlab.pipeline.surface(i)
	#mlab.colorbar(orientation='vertical')
	mlab.show()
	
	
	
	#img_ret = blend(img_target, img_source, img_mask, offset=(int(img_target.shape[0]/2 - 3*d/4),0))
	#img_ret = PIL.Image.fromarray(np.uint8(img_ret))
	#img_ret.save('./testimages/test4_ret.png')