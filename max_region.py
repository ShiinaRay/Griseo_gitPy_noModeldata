import numpy as np
import cv2
def select_max_region(imagepath):
    img = cv2.imread(imagepath)
    img_reverse = ~img
    gray = cv2.cvtColor(img_reverse, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)[1]
    nums, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # print(nums)
    # print(labels)
    # print(stats)
    # print(centroids)
    # print(stats.shape)
    # print(stats.shape[0])
    background = 0
    for row in range(stats.shape[0]):
        if stats[row, :][0] == 0 and stats[row, :][1] == 0:
            background = row
            # print(background)
    stats_no_bg = np.delete(stats, background, axis=0)
    max_idx = 0
    if len(stats_no_bg) > 0:
        max_idx = stats_no_bg[:, 4].argmax()
        # print(stats_no_bg[:, 4])
        # print(max_idx)
    max_region = np.where(labels==max_idx+1, 1, 0)
    max_region = np.array(max_region, dtype=np.uint8)
    max_region_reverse = ~(max_region * 255)
    return max_region_reverse

def main():
    # for k in range(0, 50):
    #     img = cv2.imread('./cut/'+str(k)+'.png')
    #     img_reverse = ~img
    #     # cv2.imshow('img_reverse' ,img_reverse)
    #     # cv2.waitKey()
    #     gray = cv2.cvtColor(img_reverse, cv2.COLOR_BGR2GRAY)
    #     mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)[1]
    #     # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 7)
    #     # cv2.imshow("binary", binary)
    #     # cv2.waitKey(0)
    #     mask = select_max_region(mask)
    #     mask = np.array(mask, dtype=np.uint8)
    #     # cv2.imshow("mask", mask * 255)
    #     # cv2.waitKey(0)
    #     cv2.imwrite('./test2/'+str(k)+'.png', mask * 255)
    #     mask1 = ~(mask * 255)
    #     # mask1 = np.array(mask1, dtype=np.uint8)
    #     # cv2.imshow("mask1", mask1)
    #     # cv2.waitKey(0)
    #     cv2.imwrite('./test2/'+str(k+50)+'.png', mask1)
    #     cv2.imwrite('./max_region/'+str(k)+'.png', mask1)
    for k in range(0, 1):
        max_region = select_max_region('./cut/'+str(k)+'.png')
        # cv2.imwrite('./max_region/'+str(k)+'.png', max_region)
        cv2.imshow("mask", max_region)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()



