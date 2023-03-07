
import cv2

###清除黑色背景
def Clearframe(imagepath):
    image = cv2.imread(imagepath, 1)
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 灰度图
    # cv2.imshow("gray_img)", gray_img)
    # cv2.waitKey(0)
    height, width = gray_img.shape  # 获取图片宽高
    # (_, blur_img) = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)  # 二值化 固定阈值127
    # cv2.imshow("blur_img", blur_img)
    # cv2.waitKey(0)
    # 去除黑色背景，seedPoint代表初始种子，进行四次，即对四个角都做一次，可去除最外围的黑边
    blur_img = cv2.floodFill(gray_img, mask=None, seedPoint=(0, 0), newVal=(255, 255, 255))[1]
    blur_img = cv2.floodFill(blur_img, mask=None, seedPoint=(0, height-1), newVal=(255, 255, 255))[1]
    # blur_img = cv2.floodFill(blur_img, mask=None, seedPoint=(0, 10), newVal=(255, 255, 255))[1]
    blur_img = cv2.floodFill(blur_img, mask=None, seedPoint=(width - 1, height - 1), newVal=(255, 255, 255))[1]
    blur_img = cv2.floodFill(blur_img, mask=None, seedPoint=(width - 1, 0), newVal=(255, 255, 255))[1]
    cv2.imshow("blur_img", blur_img)
    cv2.waitKey(0)
    # cv2.imwrite("./optimize/" +'0'+ ".png", blur_img)
    return blur_img

def main():
    for i in range(0,50):
        Clearframe('./cut/'+str(i)+'.png')
    # Clearframe('./cut/0.png')
    # Clearframe('./cut/1.png')
    # Clearframe('./cut/21.png')
    # ClearBackGround('./cut/2.png')
    # ClearBackGround('./cut/5.png')
    # ClearBackGround('./cut/6.png')


if __name__ == '__main__':
    main()