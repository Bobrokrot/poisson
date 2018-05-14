from tvtk.api import tvtk
import numpy as np
from mayavi import mlab
#from medpy.io import load
import nrrd
crop = 200
#image_data, image_header = nrrd.read('Berea.nhdr')
#image_data = image_data[:crop, :crop, :crop]
"""
import struct
decoded = []
with open("C1.raw","rb") as f:
	while True:
		buf = f.read(1)
		if not buf:
			break
		value = struct.unpack('<B', buf)
		decoded.append(value)


image_data = np.array(decoded).reshape((400,400,400))
"""
#X, Y, Z = np.mgrid[-10:10:100j, -10:10:100j, -10:10:100j]
#data = np.sin(X*Y*Z)/(X*Y*Z)
image_data = np.fromfile('Image/Berea.raw', dtype='<B').reshape(400,400,400)[:crop,:crop,:crop]


i = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0))
i.point_data.scalars = image_data.ravel()
i.point_data.scalars.name = 'scalars'
i.dimensions = image_data.shape #(image_data.shape[2], image_data.shape[1], image_data.shape[0])
mlab.pipeline.surface(i, color = (1,0,0))
mlab.colorbar(orientation='vertical')
mlab.show()