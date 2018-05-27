#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse
import PIL.Image
import cv2
import pyamg
import time

# pre-process the mask array so that uint64 types from opencv.imread can be adapted
def prepare_mask(mask):
	if type(mask[0][0]) is np.ndarray:
		result = np.ndarray((mask.shape[0], mask.shape[1]), dtype=np.uint8)
		for i in range(mask.shape[0]):
			for j in range(mask.shape[1]):
				if sum(mask[i][j]) > 0:
					result[i][j] = 1
				else:
					result[i][j] = 0
		mask = result
	return mask

def blend(img_target, img_source, img_mask, offset=(0, 0), mode = 0):
	# modes:
	#	0 — standard
	#	1 — mixing
	#	2 — average
	# compute regions to be blended
	region_source = (
			max(-offset[0], 0),
			max(-offset[1], 0),
			min(img_target.shape[0]-offset[0], img_source.shape[0]),
			min(img_target.shape[1]-offset[1], img_source.shape[1]))
	region_target = (
			max(offset[0], 0),
			max(offset[1], 0),
			min(img_target.shape[0], img_source.shape[0]+offset[0]),
			min(img_target.shape[1], img_source.shape[1]+offset[1]))
	region_size = (region_source[2]-region_source[0], region_source[3]-region_source[1])

	# clip and normalize mask image
	img_mask = img_mask[region_source[0]:region_source[2], region_source[1]:region_source[3]]
	
	img_mask = prepare_mask(img_mask)
	#img_mask[img_mask==0] = False
	#img_mask[img_mask!=False] = True

	# create coefficient matrix
	A = scipy.sparse.identity(np.prod(region_size), format='lil')
	for y in range(region_size[0]):
		for x in range(region_size[1]):
			if img_mask[y,x] != 0 :
				index = x+y*region_size[1]
				Np = 0
				if index+1 < np.prod(region_size):
					A[index, index+1] = -1
					Np+=1
				if index-1 >= 0:
					A[index, index-1] = -1
					Np+=1
				if index+region_size[1] < np.prod(region_size):
					A[index, index+region_size[1]] = -1
					Np+=1
				if index-region_size[1] >= 0:
					A[index, index-region_size[1]] = -1
					Np+=1
				A[index, index] = Np
	A = A.tocsr()
	
	
	
	# create poisson matrix for b
	P = pyamg.gallery.poisson(img_mask.shape)
	
	# for each layer (ex. RGB)
	N = 1 if len(img_target.shape) < 3 else img_target.shape[2]
	
	for num_layer in range(N):
		# get subimages
		if N == 1:
			tar = img_target[region_target[0]:region_target[2],region_target[1]:region_target[3]]
			sour = img_source[region_source[0]:region_source[2], region_source[1]:region_source[3]]
		else:	
			tar = img_target[region_target[0]:region_target[2],region_target[1]:region_target[3],num_layer]
			sour = img_source[region_source[0]:region_source[2], region_source[1]:region_source[3],num_layer]
		t = tar.flatten()
		s = sour.flatten()

		# create b
		
		B = P * s
		for y in range(region_size[0]):
			for x in range(region_size[1]):
				if img_mask[y,x] == 0:
					index = x+y*region_size[1]
					#B[index] = t[index]
		
		#b = scipy.sparse.lil_matrix((np.prod(region_size)))
		b = np.zeros(np.prod(region_size))
		for y in range(region_size[0]):
			for x in range(region_size[1]):
				index = x+y*region_size[1]
				if img_mask[y,x] != 0:
					sum = 0
					neighbors = [[1,0],[0,1],[0,-1],[-1,0]]
					checkIfOutOfRange = lambda ind: ind[0] >= 0 and ind[1] >= 0 and ind[0] < region_size[0] and ind[1] < region_size[1]
					for nb in neighbors:
						p2 = tuple(np.array([y,x]) + np.array(nb))
						if checkIfOutOfRange(p2) and img_mask[p2] != 0:
							s1 = int(sour[y,x]) - int(sour[p2])
							s2 = int(tar[y,x]) - int(tar[p2])
							if mode == 0: #standard
								sum+= s1
							elif mode == 1: #mixing
								sum+=  s1 if abs(s1) >= abs(s2) else s2
							elif mode == 2: #average
								sum += (1.0*(s1 + s2))/2
							elif mode == -1:
								pass
					b[index] = sum
				else:
					pass
					b[index] = t[index]
					
		# solve Ax = b
		x = pyamg.solve(A,b,verb=False,tol=1e-10)
		#print time.time() - start
		#print region_target
		#x = seidel(A,b, 1e-10)

		# assign x to target image
		x = np.reshape(x, region_size)
		x[x>255] = 255
		x[x<0] = 0
		x = np.array(x, img_target.dtype)
		#print x.shape
		if N == 1:
			img_target[region_target[0]:region_target[2],region_target[1]:region_target[3]] = x
		else:
			img_target[region_target[0]:region_target[2],region_target[1]:region_target[3],num_layer] = x
			

	return img_target

def seidel(A, b, eps):
	n = len(b)
	x = [.0 for i in range(n)]

	converge = False
	while not converge:
		x_new = np.copy(x)
		for i in range(n):
			s1 = sum(A[i,j] * x_new[j] for j in range(i))
			s2 = sum(A[i,j] * x[j] for j in range(i + 1, n))
			x_new[i] = (b[i] - s1 - s2) / A[i,i]
		
		c = sqrt(sum((x_new[i] - x[i]) ** 2 for i in range(n)))
		converge = c <= eps
		x = x_new

	return x

	
def test2():
	d = 300
	img1 = cv2.imread('./testimages/B6_10_0001.rec.16bit.tif', 0)
	img2 = cv2.imread('./testimages/B6_10_0147.rec.16bit.tif', 0)
	imgs = img1[-d:, :]
	imgs[-1, :] = img2[0, :]
	cv2.imwrite('./testimages/test2_source.png', imgs)
	imgt = np.zeros((img1.shape[0] + img2.shape[0] - d, img1.shape[1]))
	imgt[:img1.shape[0] - d, :] = img1[:-d, :]
	imgt[img1.shape[0] - d:, :] = img2
	img_mask = np.zeros(imgs.shape)
	img_mask.flags.writeable = True
	#img_mask[img1.shape[0] - d : img1.shape[0], :] = 255
	img_mask[ 1:-1 , :] = 255
	cv2.imwrite('./testimages/test2_mask.png', img_mask)
	img_source = np.asarray(imgs)
	img_source.flags.writeable = True
	img_target = np.asarray(imgt)
	img_target.flags.writeable = True
	cv2.imwrite('./testimages/test2_ret2.png', img_target)
	img_ret = blend(img_target, img_source, img_mask, offset=(img1.shape[0] - d,0))
	cv2.imwrite('./testimages/test2_ret.png', img_ret)
	
def test(mode = 0):
	d = 50; n = 5
	sufx = ['', '_mixed', '_average', '_none']
	fmask = './testimages/test' + str(n) + '_mask.png'
	fsource = './testimages/test' + str(n) + '_source.png'
	ftarget = './testimages/test' + str(n) + '_target.png'
	fret = './testimages/test' + str(n) + '_ret' + sufx[mode] + '.png'
	#img_mask = np.asarray(PIL.Image.open('./testimages/test1_mask.png'))
	img_mask = np.asarray(PIL.Image.open(fmask))
	img_mask.flags.writeable = True
	#img_source = np.asarray(PIL.Image.open('./testimages/test1_src.png'))
	img_source = np.asarray(PIL.Image.open(fsource))
	img_source.flags.writeable = True
	#img_target = np.asarray(PIL.Image.open('./testimages/test1_target.png'))
	img_target = np.asarray(PIL.Image.open(ftarget))
	img_target.flags.writeable = True
	#of = (0,int(img_target.shape[1]/2 - 3*d/4))
	of = (0,int((img_target.shape[1] - d)/2 - 1))
	#of = (40,-30)
	img_ret = blend(img_target, img_source, img_mask, offset=of, mode = mode)
	img_ret = PIL.Image.fromarray(np.uint8(img_ret))
	img_ret.save(fret)


if __name__ == '__main__':
	test(1)
