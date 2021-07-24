import cv2
import segmentation_models_pytorch as smp

img = cv2.imread("aa/Images/1.jpg",cv2.IMREAD_GRAYSCALE)
count = 0
# for x in range(512):
#     for y in range(512):
#         if img[x][y][2]!=0:
#             count=count+1
#             print(img[x][y][2])
print(img[256][256])
cv2.imshow("q", img)
cv2.waitKey()
cv2.destroyAllWindows()
model = smp.Unet()