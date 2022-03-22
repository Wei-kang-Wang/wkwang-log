import numpy as np
import os
import imageio

def translation(path, index):
	target = np.loadtxt(os.path.join(path, 'render') + str(index) + '.csv', dtype=np.string_)
	res = np.zeros((8,4))
	for i in range(8):
		a = target[i]
		a1, a2, a3, a4 = a[:24], a[25:49], a[50:74], a[75:99]
		a1 = float(a1[:4]) * 10 ** int(a1[-2:])
		a2 = float(a2[:4]) * 10 ** int(a2[-2:])
		a3 = float(a3[:4]) * 10 ** int(a3[-2:])
		a4 = int(float(a4[:4]) * 10 ** int(a4[-2:]))
		res[i,0] = a1
		res[i,1] = a2
		res[i,2] = a3
		res[i,3] = a4
	res[:,0] = np.int16(res[:,0])
	res[:,1] = np.int16(res[:,1])
	res[:,3] = np.int16(res[:,3])
	return res

def translation_short(path, index):
	target = np.loadtxt(os.path.join(path, 'render') + str(index) + '.csv', dtype=np.string_)
	res = np.zeros((8,4))
	for i in range(8):
		a = target[i]
		a1, a2, a3, a4 = a[:8], a[9:17], a[18:26], a[27:35]
		a1 = float(a1[:4]) * 10 ** int(a1[-2:])
		a2 = float(a2[:4]) * 10 ** int(a2[-2:])
		a3 = float(a3[:4]) * 10 ** int(a3[-2:])
		a4 = int(float(a4[:4]) * 10 ** int(a4[-2:]))
		res[i,0] = a1
		res[i,1] = a2
		res[i,2] = a3
		res[i,3] = a4
	res[:,0] = np.int16(res[:,0])
	res[:,1] = np.int16(res[:,1])
	res[:,3] = np.int16(res[:,3])
	return res

def calculate_mean_and_std(data_location1, data_location2, data_location3):
	result1 = np.zeros((224, 224))
	result2 = np.zeros((224, 224))
	result3 = np.zeros((224, 224))
	file_list1 = os.listdir(data_location1)
	file_list2 = os.listdir(data_location2)
	file_list3 = os.listdir(data_location3)
	for _, file in enumerate(file_list1):
		img = imageio.imread(os.path.join(data_location1, file))[:,:,:3]
		result1 += np.array(img[:,:,0])
		result2 += np.array(img[:,:,1])
		result3 += np.array(img[:,:,2])

	for _, file in enumerate(file_list2):
		img = imageio.imread(os.path.join(data_location2, file))[:,:,:3]
		result1 += np.array(img[:,:,0])
		result2 += np.array(img[:,:,1])
		result3 += np.array(img[:,:,2])

	for _, file in enumerate(file_list3):
		img = imageio.imread(os.path.join(data_location3, file))[:,:,:3]
		result1 += np.array(img[:,:,0])
		result2 += np.array(img[:,:,1])
		result3 += np.array(img[:,:,2])

	return ([result1.mean(), result2.mean(), result3.mean()], [result1.std(), result2.std(), result3.std()])
