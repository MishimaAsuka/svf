# -*- coding:utf-8 -*-
#
# envs:
#   python>=3.6
#   requirements:
#	  opencv-contrib-python==3.4.2.16
#     pyquery==1.4.0
#     numpy==1.16.2
#     labelImg==1.8.3
#

import cv2
import math
import collections
import numpy as np
from pyquery import PyQuery

def theta(X, Y):
	if X == 0:
		if Y >= 0:
			return 90
		else:
			return 270
	elif X > 0:
		return math.atan2(X, Y) / math.pi
	else:
		if Y >= 0:
			return math.atan2(X, Y) / math.pi + 180
		else:
			return math.atan2(X, Y) / math.pi - 180

def spaceValidate(pa0, pa1, pb0, pb1):
	angle1 = pa0.angle - pa1.angle
	angle2 = pb0.angle - pb1.angle
	X1 = pa1.pt[1] - pa0.pt[1]
	Y1 = pa1.pt[0] - pa0.pt[0]
	theta1 = theta(X1, Y1) - pa0.angle
	X2 = pb1.pt[1] - pb0.pt[1]
	Y2 = pb1.pt[0] - pb0.pt[0]
	theta2 = theta(X2, Y2) - pb0.angle
	if abs(angle1 - angle2) < 10 and abs(theta1 - theta2) < 10:
		return 1
	return 0

def svf(kpts_src, kpts_dst, matches):
	numRawMatches = len(matches)
	brother_matrix = np.zeros((numRawMatches, numRawMatches), dtype=np.int)

	for i in range(numRawMatches):
		for j in range(i + 1, numRawMatches):
			lp0 = kpts_src[matches[i].queryIdx]
			rp0 = kpts_src[matches[i].queryIdx]
			lp1 = kpts_dst[matches[j].trainIdx]
			rp1 = kpts_dst[matches[j].trainIdx]
			brother_matrix[i][j] = brother_matrix[j][i] = spaceValidate(lp0, rp0, lp1, rp1)

	map_size = numRawMatches
	mapId = list(range(numRawMatches))

	result = []
	while 1:
		maxv = -1
		maxid = 0
		i = 0
		while i < map_size:
			n = 0
			j = 0
			while j < map_size:
				n += brother_matrix[mapId[i]][mapId[j]]
				j += 1
			if n > maxv:
				maxv = n
				maxid = mapId[i]
			i += 1
		if maxv == 0:
			break
		result.append(matches[maxid])
		l = 0
		k = 0
		while k < map_size:
			if brother_matrix[maxid][mapId[k]]:
				mapId[l] = mapId[k]
				l += 1
			k += 1
		map_size = maxv

	return result

def sift_match(src, dst):
	""" 
		主要用来做整张截图的两两间匹配，可以准确判别出是否处在同一个UI界面下
		特征匹配同时利用了局部特征和全局特征，抵抗干扰（形变、局部变化、分辨率适配）
	"""
	detector = cv2.xfeatures2d.SIFT_create()
	src_img = cv2.imread(src)
	dst_img = cv2.imread(dst)
	kpts_src, desc_src = detector.detectAndCompute(src_img, None)
	kpts_dst, desc_dst = detector.detectAndCompute(dst_img, None)

	flann = cv2.FlannBasedMatcher()
	matches = flann.knnMatch(desc_src, desc_dst, 2)

	matches = filter(lambda m: m[0].distance < m[1].distance * 0.8, matches)
	matches = [m[0] for m in matches]

	img = cv2.drawMatches(src_img, kpts_src, dst_img, kpts_dst, matches, None)
	cv2.imshow("MATCH", img)

	result = svf(kpts_src, kpts_dst, matches)
	nimg = cv2.drawMatches(src_img, kpts_src, dst_img, kpts_dst, result, None)
	cv2.imshow("MATCH2", nimg)

def box_sift_match(src, dst, xml, name):
	""" 
		主要用来做局部的特征匹配
		需要用到全局特征来筛选掉误匹配点，所以要配合标记工具制造的xml使用
		这里选的是labelImg输出的xml，查看的话用在labelImg里打开timg.png试试
	"""
	detector = cv2.xfeatures2d.SIFT_create()
	src_img = cv2.imread(src)
	dst_img = cv2.imread(dst)
	kpts_src, desc_src = detector.detectAndCompute(src_img, None)
	kpts_dst, desc_dst = detector.detectAndCompute(dst_img, None)

	flann = cv2.FlannBasedMatcher()
	matches = flann.knnMatch(desc_src, desc_dst, 2)

	matches = filter(lambda m: m[0].distance < m[1].distance * 0.8, matches)
	matches = [m[0] for m in matches]

	result = svf(kpts_src, kpts_dst, matches)
	m0 = result[0]

	qry = PyQuery(filename=xml)
	obj = qry("object:contains(%s)" % name).eq(0)
	xmin = int(obj("xmin").text())
	xmax = int(obj("xmax").text())
	ymin = int(obj("ymin").text())
	ymax = int(obj("ymax").text())

	img = src_img[ ymin:ymax, xmin:xmax]
	kpts, decs = detector.detectAndCompute(img, None)
	matches = flann.knnMatch(decs, desc_dst, 1)

	svf_matches = []
	for m in matches:
		lp0 = kpts[m[0].queryIdx]
		rp0 = kpts_src[m0.queryIdx]
		lp1 = kpts_dst[m[0].trainIdx]
		rp1 = kpts_dst[m0.trainIdx]
		lp = collections.namedtuple("Point", ["pt", "angle"])
		lp.angle = lp0.angle
		lp.pt = (lp0.pt[0] + xmin, lp0.pt[1] + ymin)
		if spaceValidate(lp, rp0, lp1, rp1):
			svf_matches.append(m[0])

	return img, kpts, dst_img, kpts_dst, svf_matches


if __name__ == "__main__":
	"""
	需要使用labelImg进行图片box的标注
	附带范例中已标注的标签：
	- 精英化
	- 潜能
	- 返回
	- 关系图
	- 干员
	- 换装
	- 信息
	- 等级经验
	- 天赋
	"""
	src = "timg.jpeg"
	dst = "timg_1.jpeg"
	xml = "timg.xml"
	name = "信息"

	img, kpts, dst_img, kpts_dst, svf_matches = box_sift_match(src, dst, xml, name)
	
	x = np.median([kpts_dst[m.trainIdx].pt[0] for m in svf_matches])
	y = np.median([kpts_dst[m.trainIdx].pt[1] for m in svf_matches])

	print("'%s'对应的坐标位置位%d,%d" % (name, x, y))

	cv2.circle(dst_img, (int(x), int(y)), 20, (255, 0, 0), -1)
	r = cv2.drawMatches(img, kpts, dst_img, kpts_dst, svf_matches, None)
	cv2.imshow("MATCH_BOX", r)
	cv2.waitKey(0)