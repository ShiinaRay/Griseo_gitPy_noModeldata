import cv2
import numpy as np

threshold = 40 # 阈值


# gray = cv2.imread('./0.png') # 导入图片
gray = cv2.imread('./cut/49.png') # 导入图片
gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY) # 转换为灰度图像
#
# binary1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 7)
# cv2.imshow('binary1',binary1) # 效果展示
# cv2.waitKey()
#
# binary2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11)
# cv2.imshow('binary2',binary2) # 效果展示
# cv2.waitKey()
# # binary22 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 5)
# # cv2.imshow('binary22',binary22) # 效果展示
# # cv2.waitKey()
# binary3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 3)
# cv2.imshow('binary3',binary3) # 效果展示
# cv2.waitKey()
#
# (_, blur_img) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # 二值化 固定阈值127
# cv2.imshow('blur_img',blur_img) # 效果展示
# cv2.waitKey()
#
# (_, blur_img1) = cv2.threshold(gray, 168, 255, cv2.THRESH_BINARY)  # 二值化 固定阈值127
# cv2.imshow('blur_img1',blur_img1) # 效果展示
# cv2.waitKey()
#
nrow = gray.shape[0] # 获取图片尺寸
ncol = gray.shape[1]

rowc = gray[:,int(1/2*nrow)] # 无法区分黑色区域超过一半的情况
colc = gray[int(1/2*ncol),:]

rowflag = np.argwhere(rowc > threshold)
colflag = np.argwhere(colc > threshold)

left,bottom,right,top = rowflag[0,0],colflag[-1,0],rowflag[-1,0],colflag[0,0]

cv2.imshow('name',gray[left:right,top:bottom]) # 效果展示
cv2.waitKey()

