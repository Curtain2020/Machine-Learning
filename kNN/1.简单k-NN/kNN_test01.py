# -*- coding: UTF-8 -*-
import numpy as np
import operator
import collections

"""
函数说明:创建数据集

Parameters:
	无
Returns:
	group - 数据集
	labels - 分类标签
Modify:
	2017-07-13
"""
def createDataSet():
	#四组二维特征
	group = np.array([[1,101],[5,89],[108,5],[115,8]])
	#四组特征的标签
	labels = ['爱情片','爱情片','动作片','动作片']
	return group, labels

"""
函数说明:kNN算法,分类器

Parameters:
	inX - 用于分类的数据(测试集)
	dataSet - 用于训练的数据(训练集)
	labes - 分类标签
	k - kNN算法参数,选择距离最小的k个点
Returns:
	sortedClassCount[0][0] - 分类结果

Modify:
	2017-11-09 by Cugtyt 
		* GitHub(https://github.com/Cugtyt) 
		* Email(cugtyt@qq.com)
		Use list comprehension and Counter to simplify code
	2017-07-13
"""
def classify0(inx, dataset, labels, k):
	# 计算距离
	dist = np.sum((inx - dataset)**2, axis=1)**0.5
	#对上述结果沿着第二个轴（即列）进行求和，得到一个一维数组
	# k个最近的标签
	k_labels = [labels[index] for index in dist.argsort()[0 : k]]
	#.argsort()函数返回数组从小到大排序的索引值，所以.argsort()[0 : k]表示返回数组中前k个最小值的索引。
	# 出现次数最多的标签即为最终类别
	label = collections.Counter(k_labels).most_common(1)[0][0]
	#.Counter(k_labels)创建一个计数器对象，统计k_labels中各个元素的出现次数。
	#.most_common(1)返回出现次数最多的元素及其出现次数，以列表的形式返回。所以.most_common(1)[0][0]表示返回出现次数最多的元素。
	return label

if __name__ == '__main__':
	#创建数据集
	group, labels = createDataSet()
	#测试集
	test = [101,20]
	#kNN分类
	test_class = classify0(test, group, labels, 3)
	#打印分类结果
	print(test_class)
