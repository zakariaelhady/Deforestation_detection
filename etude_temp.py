import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def difference(ng1,ng2):
    Gaus = cv.GaussianBlur(ng1,(5,5),0) 
    Seuil1,bin1 = cv.threshold(Gaus,0,255,cv.THRESH_OTSU) 
    Gaus = cv.GaussianBlur(ng2,(5,5),0)
    Seuil2,bin2 = cv.threshold(Gaus,0,255,cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(15,15)) #choisir l'element structurant
    imFermeture1 = cv.morphologyEx(bin1, cv.MORPH_CLOSE, kernel)
    imFermeture2 = cv.morphologyEx(bin2, cv.MORPH_CLOSE, kernel)
    #difference entre les deux images
    Diff = np.bitwise_xor(imFermeture1, imFermeture2)
    return Diff

ng_2000 = cv.imread('images/2000.jpg',0) 
ng_2001 = cv.imread('images/2001.jpg',0) 
ng_2002 = cv.imread('images/2002.jpg',0) 
ng_2003 = cv.imread('images/2003.jpg',0) 
ng_2004 = cv.imread('images/2004.jpg',0) 
ng_2005 = cv.imread('images/2005.jpg',0) 
ng_2006 = cv.imread('images/2006.jpg',0) 
ng_2007 = cv.imread('images/2007.jpg',0) 
ng_2008 = cv.imread('images/2008.jpg',0)
ng_2009 = cv.imread('images/2009.jpg',0) 
ng_2010 = cv.imread('images/2010.jpg',0) 
ng_2011 = cv.imread('images/2011.jpg',0) 
ng_2012 = cv.imread('images/2012.jpg',0)
ngs=[ng_2000,ng_2001,ng_2002,ng_2003,ng_2004,ng_2005,ng_2006,ng_2007,ng_2008,ng_2009,ng_2010,ng_2011,ng_2012]

diffs=[]

for i in range(0,len(ngs)-1):
    diffs.append(difference(ngs[i],ngs[i+1]))
    

years=['2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012']
pourcentages=[]
for i in range(0,len(diffs)):
    total_couverture_diff= np.sum(diffs[i])/255
    pourcentages.append(float('{0:.2f}'.format(total_couverture_diff/diffs[i].size*100)))
    print(f'zones déforestées entre {years[i]} et {years[i+1]}: {pourcentages[i]}')

x=[]
for i in range(0,len(years)-1):
    x.append(f'{years[i]}-{years[i+1]}')
y = pourcentages

plt.plot(x, y)
plt.xlabel('années')
plt.ylabel('pourcentages')
plt.title('étude temporelle de la déforestation')

plt.show(block=False)




rgb_2001 = cv.imread('images/2001.jpg') 
rgb_2002 = cv.imread('images/2002.jpg') 
rgb_2003 = cv.imread('images/2003.jpg') 
rgb_2004 = cv.imread('images/2004.jpg') 
rgb_2005 = cv.imread('images/2005.jpg') 
rgb_2006 = cv.imread('images/2006.jpg') 
rgb_2007 = cv.imread('images/2007.jpg') 
rgb_2008 = cv.imread('images/2008.jpg')
rgb_2009 = cv.imread('images/2009.jpg') 
rgb_2010 = cv.imread('images/2010.jpg') 
rgb_2011 = cv.imread('images/2011.jpg') 
rgb_2012 = cv.imread('images/2012.jpg')
rgbs=[rgb_2001,rgb_2002,rgb_2003,rgb_2004,rgb_2005,rgb_2006,rgb_2007,rgb_2008,rgb_2009,rgb_2010,rgb_2011,rgb_2012]

def color_diff(img,diff_img):
    for i in range(0,diff_img.shape[0]):
        for j in range(0,diff_img.shape[1]):
            if(diff_img[i][j]!=0):
                img[i][j]=np.array([255,0,0])
    return img



plt.figure()
for i in range(0,9):
    plt.subplot(331+i)
    img=color_diff(rgbs[i],diffs[i])
    plt.imshow(img)
    plt.title(f'{x[i]}')
    plt.show(block=False)


plt.figure()
for i in range(0,2):
    plt.subplot(221+i)
    img=color_diff(rgbs[i+9],diffs[i+9])
    plt.imshow(img)
    plt.title(f'{x[i+9]}')
    plt.show(block=False)


i=2
plt.subplot(221+i)
img=color_diff(rgbs[i+9],diffs[i+9])
plt.imshow(img)
plt.title(f'{x[i+9]}')
plt.show()