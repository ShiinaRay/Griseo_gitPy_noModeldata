# -*- coding: utf-8 -*-

import cv2
import numpy as np
import pytesseract
import Lenet5Detection
import os
import max_region

def cutimage(filepath):

    # image = cv2.imread('test.jpg', 1)
    image = cv2.imread(filepath, 1)
    # 灰度图片
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 二值化
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -5)
    # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 7)
    # ret,binary = cv2.threshold(~gray, 127, 255, cv2.THRESH_BINARY)
    # cv2.imshow("二值化图片：", binary)  # 展示图片
    # cv2.waitKey(0)

    # mybinary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 7)
    # cv2.imwrite("./uploads/mybinary.png", mybinary)  # 保存图像文件


    rows, cols = binary.shape
    scale = 40
    # 识别横线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
    eroded = cv2.erode(binary, kernel, iterations=1)
    # cv2.imshow("Eroded Image",eroded)
    dilatedcol = cv2.dilate(eroded, kernel, iterations=1)
    # cv2.imshow("表格横线展示：", dilatedcol)
    # cv2.waitKey(0)

    # 识别竖线
    scale = 20
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilatedrow = cv2.dilate(eroded, kernel, iterations=1)
    # cv2.imshow("表格竖线展示：", dilatedrow)
    # cv2.waitKey(0)

    # 标识交点
    bitwiseAnd = cv2.bitwise_and(dilatedcol, dilatedrow)
    # cv2.imshow("表格交点展示：", bitwiseAnd)
    # cv2.waitKey(0)
    # cv2.imwrite("my.png",bitwiseAnd) #将二值像素点生成图片保存
    # print(bitwiseAnd)

    # 标识表格
    merge = cv2.add(dilatedcol, dilatedrow)
    # cv2.imshow("表格整体展示：", merge)
    # cv2.waitKey(0)

    # 两张图片进行减法运算，去掉表格框线
    merge2 = cv2.subtract(binary, merge)
    # cv2.imshow("图片去掉表格框线展示：", merge2)
    # cv2.waitKey(0)


    # 识别黑白图中的白色交叉点，将横纵坐标取出
    ys, xs = np.where(bitwiseAnd > 0)

    mylisty = []  # 纵坐标
    mylistx = []  # 横坐标

    # 通过排序，获取跳变的x和y的值，说明是交点，否则交点会有好多像素值值相近，我只取相近值的最后一点
    # 这个10的跳变不是固定的，根据不同的图片会有微调，基本上为单元格表格的高度（y坐标跳变）和长度（x坐标跳变）
    i = 0
    # print(xs)
    myxs = np.sort(xs)
    # print(myxs)
    for i in range(len(myxs) - 1):
        if (myxs[i + 1] - myxs[i] > 15):
            mylistx.append(myxs[i])
        i = i + 1
    mylistx.append(myxs[i])  # 要将最后一个点加入

    i = 0
    myys = np.sort(ys)
    # print(np.sort(ys))
    for i in range(len(myys) - 1):
        if (myys[i + 1] - myys[i] > 15):
            mylisty.append(myys[i])
        i = i + 1
    mylisty.append(myys[i])  # 要将最后一个点加入

    # print('mylisty', mylisty)
    # print('mylistx', mylistx)
    # print('开始保存分割后的图像')
    # 循环y坐标，x坐标分割表格
    # a = []
    # for i in range(len(mylisty) - 1):
    #     a.append([])
    #     for j in range(len(mylistx) - 1):
    #         # 在分割时，第一个参数为y坐标，第二个参数为x坐标
    #         ROI = image[mylisty[i] + 3:mylisty[i + 1] - 3, mylistx[j]:mylistx[j + 1] - 3]  # 减去3的原因是由于我缩小ROI范围
    #         cv2.imshow("分割后子图片展示：", ROI)
    #         cv2.waitKey(0)
    #         a[i].append(mylistx[j])


    count = 0
    draw_marker = image
    results1 = []
    results2 = []
    answer1 = [7, 6, 0, -1, -1, -1, 0, -1, 5, 9, 1, -1, 0, -1, 6, 3, -1, -1, 0, -1, 8, 9, -1, -1, 6, -1, 8, 3, -1, -1]
    answer2 = [4, 6, 8, 6, 5, 0, 0, 4, 7, 3, 5, -1, 6, 0, 1, -1, 6, 4, 1, -1]
    #修改 mylist值  来适应右边的表格
    # for n in range(-1, -6, -1):
    #     mylisty[n]=mylisty[n]-6
    mylisty[-1] = mylisty[-1] - 5
    mylisty[-2] = mylisty[-2] - 5+3
    mylisty[-3] = mylisty[-3] - 5+7
    mylisty[-4] = mylisty[-4] - 5
    mylisty[-5] = mylisty[-5] - 6
    mylistx[-1]=mylistx[-1]-0
    mylistx[-2]=mylistx[-2]-0
    mylistx[-3]=mylistx[-3]-0
    mylistx[-4]=mylistx[-4]-0
    mylistx[-5]=mylistx[-5]-0
    mylistx[-6]=mylistx[-6]+2
    # print('mylisty', mylisty)
    # print('mylistx', mylistx)
    i=-1
    j=-1
    k=0
    for i in range(-1, -6, -1):
        for j in range(-1, -7, -1):
            cut = merge2[mylisty[i - 1]:mylisty[i], mylistx[j - 1]:mylistx[j]]  # y    x
            cut_reverse = ~cut
            cv2.imwrite("./cut/" + str(k) + ".png", cut_reverse)  # 保存图像文件
            # results1.append(Lenet5Detection.evaluate_one_image("./cut/"+str(k)+".png"))
            #--------------------------------------optimize-----------------------------------------------------------
            max_region_image = max_region.select_max_region('./cut/' + str(k) + '.png')
            cv2.imwrite('./max_region/' + str(k) + '.png', max_region_image)
            if answer1[k] == -1:
                k=k+1
                results1.append(-1)
                continue
            else:
                # results1.append(Lenet5Detection.evaluate_one_image("./cut/"+str(k)+".png"))
                results1.append(Lenet5Detection.evaluate_one_image("./max_region/" + str(k) + ".png"))
                if answer1[k] == results1[k]:
                    draw_marker = cv2.drawMarker(draw_marker, (mylistx[j], mylisty[i]), (0, 255, 0), cv2.MARKER_DIAMOND, thickness=2)
                    cv2.imwrite("./uploads/results.png", draw_marker)  # 保存图像文件
                    count = count + 1
                else:
                    draw_marker = cv2.drawMarker(draw_marker, (mylistx[j], mylisty[i]), (0, 0, 255), cv2.MARKER_TILTED_CROSS, thickness=2)
                    cv2.imwrite("./uploads/results.png", draw_marker)  # 保存图像文件
            k=k+1

    # print(k)   k=30

    #修改 mylist值  来适应左边的表格
    # for n in range(-1, -6, -1):
    #     mylisty[n]=mylisty[n]-2
    mylisty[-1] = mylisty[-1] - 5-1
    mylisty[-2] = mylisty[-2] - 5-1-3
    mylisty[-3] = mylisty[-3] - 4-1-7
    mylisty[-4] = mylisty[-4] - 3-1
    mylisty[-5] = mylisty[-5] - 4
    mylistx[-10]=mylistx[-10]-2
    mylistx[-11]=mylistx[-11]-0
    mylistx[-12]=mylistx[-12]-7
    mylistx[-13]=mylistx[-13]-18
    # mylistx[-13]=mylistx[-13]-20      #直接-20   1题 的1对  3题的5不对 ∴先-18 后再-2
    mylistx[-14]=mylistx[-14]-50
    # print('mylisty', mylisty)
    # print('mylistx', mylistx)
    i=-1
    j=-1
    k=0
    for i in range(-1, -6, -1):
        for j in range(-10, -14, -1):
            cut = merge2[mylisty[i - 1]:mylisty[i], mylistx[j - 1]:mylistx[j]]  # y    x
            cut_reverse = ~cut
            cv2.imwrite("./cut/" + str(k+30) + ".png", cut_reverse)  # 保存图像文件
            # results2.append(Lenet5Detection.evaluate_one_image("./cut/" + str(k+30) + ".png"))
            #--------------------------------------optimize-----------------------------------------------------------
            max_region_image  = max_region.select_max_region('./cut/' + str(k+30) + '.png')
            cv2.imwrite('./max_region/' + str(k+30) + '.png', max_region_image)
            if answer2[k] == -1:
                k = k + 1
                results2.append(-1)
                continue
            else:
                # results2.append(Lenet5Detection.evaluate_one_image("./cut/" + str(k+30) + ".png"))
                results2.append(Lenet5Detection.evaluate_one_image("./max_region/" + str(k + 30) + ".png"))
                if answer2[k] == results2[k]:
                    draw_marker = cv2.drawMarker(draw_marker, (mylistx[j], mylisty[i]), (0, 255, 0), cv2.MARKER_DIAMOND,
                                                 thickness=2)
                    cv2.imwrite("./uploads/results.png", draw_marker)  # 保存图像文件
                    count = count + 1
                else:
                    draw_marker = cv2.drawMarker(draw_marker, (mylistx[j], mylisty[i]), (0, 0, 255), cv2.MARKER_TILTED_CROSS,
                                                 thickness=2)
                    cv2.imwrite("./uploads/results.png", draw_marker)  # 保存图像文件
            k=k+1
            if (k == 13): mylistx[-13] = mylistx[-13] - 2    #直接-20   1题 的1对  3题的5不对 ∴先-18 后再-2
    # print(k)  k=20
    print(results1)
    print(results2)
    scorecount = (count/32)*100
    print(scorecount)
    return scorecount



def main():
    cutimage('./test1.jpg')

if __name__ == '__main__':
    main()




    # score = 0
    # if results1[0] == answer1[0] and results1[1] == answer1[1] and results1[2] == answer1[2]:
    #     score = score + 10
    # if results1[6] == answer1[6] and results1[8] == answer1[8] and results1[9] == answer1[9] and results1[10] == answer1[10]:
    #     score = score + 10
    # if results1[12] == answer1[12] and results1[14] == answer1[14] and results1[15] == answer1[15]:
    #     score = score + 10
    # if results1[18] == answer1[18] and results1[20] == answer1[20] and results1[21] == answer1[21]:
    #     score = score + 10
    # if results1[24] == answer1[24] and results1[25] == answer1[25] and results1[26] == answer1[26] and results1[27] == answer1[27]:
    #     score = score + 10
    #
    # if results2[0] == answer2[0] and results2[1] == answer2[1] and results2[2] == answer2[2] and results2[3] == answer2[3]:
    #     score = score + 10
    # if results2[4] == answer2[4] and results2[5] == answer2[5] and results2[6] == answer2[6]:
    #     score = score + 10
    # if results2[8] == answer2[8] and results2[9] == answer2[9] and results2[10] == answer2[10]:
    #     score = score + 10
    # if results2[12] == answer2[12] and results2[13] == answer2[13]:
    #     score = score + 10
    # if results2[16] == answer2[16] and results2[17] == answer2[17] and results2[18] == answer2[18]:
    #     score = score + 10



    # imgList = os.listdir('./cut/')
    # print(imgList)
    # # 按照数字进行排序后按顺序读取文件夹下的图片
    # imgList.sort(key=lambda x: int(x.replace("", "").split('.')[0]))
    # print(imgList)
    # answer = [8, 6, 0, 4, 9, 0, 2, 0, 6, 6, 0, 4, 1, 1, 1, 8, 2, 0, 4, 7, 5, 9, 7, 9, 9, 4, 5, 2, 4, 1, 0, 1]
    # results = []
    # for count in range(0, len(imgList)):
    #     im_name = imgList[count]
    #     im_path = os.path.join('./cut/', im_name)
    #     print(im_path)
    #     results.append(Lenet5Detection.evaluate_one_image(im_path))
    #
    # index_to_delete = [3, 4, 5, 7, 11, 13, 16, 17, 19, 22, 23, 28, 29, 37, 41, 44, 45, 49]
    # for index in reversed(index_to_delete):
    #     results.pop(index)
    #
    # for i in range(0,len(results)):
    #     if results[i] == answer[i]:
    #         draw_marker = cv2.drawMarker(image, (mylistx[j], mylisty[i]), (0, 255, 0), cv2.MARKER_DIAMOND, thickness=2)
    #         cv2.imwrite("./uploads/results.png", draw_marker)  # 保存图像文件
    #     else:





    # cut = image[576:620, 830:863]  # y    x-40    6
    # cv2.imshow("分割后子图片展示：", cut)
    # cv2.waitKey(0)
    # cv2.imwrite("./cut.png", cut)  # 保存图像文件
    # # cv2.imwrite(saveFile, cut, [int(cv2.IMWRITE_PNG_COMPRESSION), 8])  # 保存图像文件, 设置压缩比为 8
    # cutgray = cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)
    # # cv2.imshow("分割后子图片灰度展示：", cutgray)
    # # cv2.waitKey(0)
    # cutbinary = cv2.adaptiveThreshold(cutgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 7)
    # cv2.imshow("分割后子图片二值化展示：", cutbinary)
    # cv2.waitKey(0)
    # cv2.imwrite("./cutbinary.png", cutbinary)  # 保存图像文件
    # results = Lenet5Detection.evaluate_one_image("./cutbinary.png")
    # if results == 6:
    #     draw_marker = cv2.drawMarker(image, (863,620 ), (0, 255, 0), cv2.MARKER_DIAMOND, thickness=2)
    #     cv2.imshow("draw_marker", draw_marker)
    #     cv2.waitKey(0)
    #     cv2.imwrite("./uploads/results.png", draw_marker)  # 保存图像文件
    # else:
    #     draw_marker = cv2.drawMarker(image, (863, 620), (255, 0, 0), cv2.MARKER_TILTED_CROSS, thickness=2)
    #     cv2.imshow("draw_marker", draw_marker)
    #     cv2.waitKey(0)
    #     cv2.imwrite("./uploads/results.png", draw_marker)  # 保存图像文件