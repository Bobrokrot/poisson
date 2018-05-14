import numpy as np
import cv2

def test2():
	#fname1 = './testimages/B6_10_0051.rec.16bit.tif'
	#fname2 = './testimages/B6_10_0147.rec.16bit.tif'
	fname1 = './testimages/Carb0006.tif'
	fname2 = './testimages/Carb0239.tif'#'./testimages/Carb1015.tif'
	d = 50
	crop = 500
	img1 = cv2.imread(fname1, 0)[-crop:, -crop:]
	img2 = cv2.imread(fname2, 0)[-crop:, -crop:]
	#img1 = cv2.imread('./testimages/frontend-large.jpg', 0)
	#img2 = cv2.imread('./testimages/frontend-large.jpg', 0)
	#imgs = np.zeros((d,img1.shape[1]))
	imgs = img1[:, -d:]
	#imgs[-1, :] = img2[0, :]
	cv2.imwrite('./testimages/test6_source.png', imgs)
	imgt = np.zeros((img1.shape[0], img1.shape[1] + img2.shape[1] - d))
	imgt[:, :img1.shape[1] - d] = img1[:, :-d]
	imgt[:, img1.shape[1] - d:] = img2
	img_mask = np.zeros(imgs.shape)
	img_mask.flags.writeable = True
	#img_mask[img1.shape[0] - d : img1.shape[0], :] = 255
	img_mask[ 1:-1 , 1:-1] = 255
	cv2.imwrite('./testimages/test6_mask.png', img_mask)
	cv2.imwrite('./testimages/test6_target.png', imgt)

if __name__ == '__main__':
	test2()