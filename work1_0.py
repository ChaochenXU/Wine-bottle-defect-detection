# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 09:58:43 2020

@author: cao sheng
"""

"""
基本说明：
1.使用image作为初始图像
2.保存image3的图像
3.image2用作数字演示
4.image1用作不保存的展示
"""

import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt

#####################################################
def showpic(image):
    cv.imshow('image_pic',image)
    cv.waitKey(0)
    cv.destroyWindow('image_pic')
    return

def showhist(image):
    image4 = image.ravel()                                              #将图像转化为一维数组
    plt.hist(image4,256)                                                #绘制直方图
    plt.show
    return
#####################################################
print("start!")

#####################################################
"""基本图像操作"""
image = cv.imread("E:\python_work\opencv_work\piclib\pic1.jpg")     #读取图片

r1 = cv.namedWindow("picWindow1")                                   #创建一个窗口
r2 = cv.imshow("picWindow1",image)                                  #在目标窗口显示图片
#####################################################
"""显示三基色通道灰度图片"""
b,g,r = cv.split(image)                                             #拆分函数
cv.imshow("b",b)
cv.imshow("g",g)
cv.imshow("r",r)
imagebgr = cv.merge([b,g,r])                                        #合并函数
cv.imshow("picWindow2",imagebgr)

"""清除该代码的窗口以更清晰显示其他窗口"""
cv.waitKey(100)
cv.destroyWindow("b")
cv.destroyWindow("g")
cv.destroyWindow("r")
cv.waitKey(100)
cv.destroyWindow("picWindow2")
#####################################################
"""图像属性获取"""
print("image.shape",image.shape)                                    #图像大小属性，返回（行数【高】，列数【宽】，通道数）
print("image.size",image.size)                                      #图像像素数目属性
print("image.dtype",image.dtype)                                    #图像类型属性
#####################################################
"""用Numpy库生成一个随机色彩图"""
image1 = np.random.randint(0,256,size = [256,256,3],dtype = np.uint8)
cv.imshow("image1",image1)
cv.waitKey(100)
cv.destroyWindow("image1")
#####################################################
"""以下请解除注释后运行（ctrl+1）"""
d1 = np.random.randint(0,256,size = [4,4],dtype = np.uint8)
d2 = np.random.randint(0,256,size = [4,4],dtype = np.uint8)
"""加减法和cv2加减法比较"""
#d3 = cv.add(d1,d2)
#d4 = cv.subtract(d1,d2)
#print("d1=\n",d1)
#print("d2=\n",d2)
#print("d1+d2=\n",d1+d2)
#print("d1+d2(add)=\n",d3)
#print("d1-d2=\n",d1-d2)
#print("d1-d2(subtract)=\n",d4)
#
#"""乘除法"""
#d3 = np.dot(d1,d2)
#d4 = cv.multiply(d1,d2)
#print("d1=\n",d1)
#print("d2=\n",d2)
#print("d1*d2(dot)=\n",d3)
#print("d1*d2(multiply)=\n",d4)
#d3 = np.divide(d1,d2)
#d4 = cv.divide(d1,d2)
#print("d1/d2(np.divide)=\n",d3)
#print("d1/d2(cv.divide)=\n",d4)
#####################################################
"""逻辑运算"""
image2 = np.zeros(image.shape,dtype = np.uint8)                     #构建掩模图像
image2[100:400,100:400] = 255
image3 = cv.bitwise_and(image,image2)                               #按位与，取出掩模内的图像(剪裁)
#image3 = cv.bitwise_or(image,image2)                                #按位或
#image3 = cv.bitwise_not(image,image2)                               #按位非
#image3 = cv.bitwise_xor(image,image2)                               #按位异或

cv.imshow("image3",image3)
cv.waitKey(100)
cv.destroyWindow("image3")
#####################################################
"""图像色彩空间的转换"""
image3 = cv.cvtColor(image,cv.COLOR_BGR2RGB)                        #转换色彩空间

cv.imshow("image3",image3)
cv.waitKey(100)
cv.destroyWindow("image3")
#####################################################
"""第四章"""
"""图像的几何变换"""
h,w = image.shape[:2]
                                                                    #设定变换矩阵(M11x + M12y + M13,M21x + M22y + M23)
M = np.float32([[1,0,120],[0,1,-120]])                              #平移的矩阵配置
imageMove = cv.warpAffine(image,M,(w,h))

cv.imshow("imageMove",imageMove)
cv.waitKey(500)
cv.destroyWindow("imageMove")

M = np.float32([[0.5,0,0],[0,0.5,0]])                               #缩放的矩阵配置
imageMove = cv.warpAffine(image,M,(w,h))

cv.imshow("imageMove",imageMove)
cv.waitKey(100)
cv.destroyWindow("imageMove")

M = cv.getRotationMatrix2D((w/3,h/3),40,0.4)                        #cv2里的仿射变换函数，元素为：中心，顺时针角度，放大倍数
imageMove = cv.warpAffine(image,M,(w,h))

cv.imshow("imageMove",imageMove)
cv.waitKey(100)
cv.destroyWindow("imageMove")
#####################################################
"""重映射变换"""
image2 = np.random.randint(0,256,size = [8,6],dtype = np.uint8)
#h,w = image.shape
h,w = image.shape[:2]

x = np.zeros((h,w),np.float32)
y = np.zeros((h,w),np.float32)

for i in range(h):
    for j in range(w):
        x.itemset((i,j),h-1-j)
        y.itemset((i,j),i)
rst = cv.remap(image,x,y,cv.INTER_LINEAR)

cv.imshow("image3",rst)
cv.waitKey(100)
cv.destroyWindow("image3")
#print("image2=\n",image2)
#print("rst=\n",rst)
#####################################################
"""投影变换"""
h,w = image.shape[:2]

src = np.array([[0,0],[w-1,0],[0,h-1],[w-1,h-1]],np.float32)        #原图像四个需要变换的像素点
dst = np.array([[80,80],[w/2,50],[80,h-80],[w-40,h-40]],np.float32) #投影变换中的四个像素点

M = cv.getPerspectiveTransform(src,dst)                             #计算出投影变换矩阵，使用cv里的函数
image3 = cv.warpPerspective(image,M,(w,h),borderValue = 125)        #进行投影变换

cv.imshow("image3",image3)
cv.waitKey(100)
cv.destroyWindow("image3")
#####################################################
"""极坐标变换"""
image3 = cv.linearPolar(image,(251,249),225,cv.INTER_LINEAR)        #极坐标变换函数（原始图像，输出图像【此处没有使用】，极坐标变换中心，极坐标的最大距离，插值算法）

cv.imshow("image3",image3)
cv.waitKey(100)
cv.destroyWindow("image3")

"""另一种极坐标方法"""
M1 = 20
M2 = 50
M3 = 90

dst1 = cv.logPolar(image,(251,249),M1,cv.WARP_FILL_OUTLIERS)        #极坐标变换函数（原始图像，输出图像，极坐标变换中心，极坐标变换系数，转换方向）
dst2 = cv.logPolar(image,(251,249),M2,cv.WARP_FILL_OUTLIERS)
dst3 = cv.logPolar(image,(251,249),M3,cv.WARP_FILL_OUTLIERS)

cv.imshow("image31",dst1)
cv.imshow("image32",dst2)
cv.imshow("image33",dst3)
cv.waitKey(100)
cv.destroyWindow("image31")
cv.destroyWindow("image32")
cv.destroyWindow("image33")
#####################################################
"""第五章：直方图处理"""
"""用plot绘制直方图，这样画出来像是折线图，下面那种是直方图""
hist = cv.calcHist([image],[0],None,[256],[0,256])                   #计算其统计直方图相关信息，参数说明：（原始图像【要用[]括起来】，指定通道编号【要用[]括起来】，表示掩模图像【不用时设置为None】，表示BINS值【要用[]括起来】，表示像素范围，累计标志【一般不用设置】）
plt.plot(hist,'r')
plt.show
"""
#####################################################
"""使用函数画图表"""
#arr1 = [1,2,3,4,5,6,7,8,9]
#arr2 = [5,4,6,9,1,3,8,7,5,6,2,4]
#plt.plot(arr1)
#plt.plot(arr2,'r')
#plt.show
#####################################################
"""用pyplot绘制直方图"""
image4 = image.ravel()                                              #将图像转化为一维数组
plt.figure("原始直方图")
plt.hist(image4,256,color='g')                                      #绘制直方图
plt.show
""""""
#####################################################
"""直方图正规化"""
imagemax = np.max(image)                                            #确定原图像的灰度级范围
imagemin = np.min(image)

min_1 = 0                                                           #确定标准化以后的灰度级范围
max_1 = 255

m = float(max_1 - min_1)/(imagemax - imagemin)                      #进行图像标准化
n = min_1 - min_1*m
image1 = m*image + n
image2 = image1
image1 = image1.astype(np.uint8)

cv.imshow("standard",image1)
cv.waitKey(100)
cv.destroyWindow("standard")

"""使用opencv中的normalize函数进行标准化"""
#image1 = cv.normalize(image,image,255,0,cv.NORM_MINMAX,cv.CV_8U)
#showpic(image1)
#showhist(image1)
#####################################################
"""直方图均衡化"""
#原理说明：将灰度级均匀化
equ = cv.equalizeHist(b)                                            #这个函数只能处理单通道灰度图像，但是考虑可以单通道处理结束后合并图像
plt.figure("均衡化直方图")
#showpic(equ)
showhist(equ)

clahe = cv.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))            #创建CLAHE对象
dst = clahe.apply(b)                                            #限制对比度的自适应阈值均衡化
plt.figure("限制均衡化直方图")
#showpic(dst)
showhist(dst)
#####################################################
"""第六章图像平滑滤波处理"""
"""高斯滤波"""
gauss = cv.GaussianBlur(image,(5,5),0,0)                        #高斯滤波，参数含义（原始图像；滤波卷积核大小；卷积核在水平方向上的权重；卷积核在竖直方向上的权重；边界值的处理方式【可省略】）。当权重为0时，计算式为0.3*[(ksize-1)*0.5-1]+0.8

cv.imshow("gauss",gauss)                                        #显示图片
cv.waitKey(100)

cv.destroyWindow("gauss")

"""均值滤波(展示3种不同大小的卷积核效果)"""
means5 = cv.blur(image,(5,5))                                   #原理是卷积核内部采取平均值方式定值，卷积核越大越糊
means10 = cv.blur(image,(10,10))
means20 = cv.blur(image,(20,20))

cv.imshow("means5",means5)
cv.imshow("means10",means10)
cv.imshow("means20",means20)
cv.waitKey(100)
cv.destroyWindow("means5")
cv.destroyWindow("means10")
cv.destroyWindow("means20")

"""方框滤波"""
box5_0 = cv.boxFilter(image,-1,(5,5),normalize=0)               #5*5卷积核，不进行归一化
box2_0 = cv.boxFilter(image,-1,(2,2),normalize=0)
box5_1 = cv.boxFilter(image,-1,(5,5),normalize=1)
box2_1 = cv.boxFilter(image,-1,(2,2),normalize=1)               #2*2卷积核，进行归一化

cv.imshow("box5_0",box5_0)
cv.imshow("box2_0",box2_0)
cv.imshow("box5_1",box5_1)
cv.imshow("box2_1",box2_1)
cv.waitKey(100)
cv.destroyWindow("box5_0")
cv.destroyWindow("box2_0")
cv.destroyWindow("box5_1")
cv.destroyWindow("box2_1")

"""中值滤波"""

#####################################################
"""检测问题""
cv.imshow("image1",image1)
cv.imshow("image2",image2)
cv.imshow("image3",image3)
cv.imshow("image4",image4)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imshow("image",image)
"""
#####################################################
r5 = cv.imwrite("E:\python_work\opencv_work\piclib\write_pic2.jpg",image3)   #保存一张图像

r3 = cv.waitKey(0)                                                  #等待按键，参数为ms，如果为0或者负数，则无限等待按键
r4 = cv.destroyAllWindows()                                         #删除所有窗口

print("end!")