import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



ng_2000 = cv.imread('amazon_2000.png',0) #lire l'image au niveau de gris (flag=0)
ng_2012 = cv.imread('amazon_2012.png',0)

Gaus = cv.GaussianBlur(ng_2000,(5,5),0) #réduire le bruit dans notre image 
Seuil1,bin_2000 = cv.threshold(Gaus,0,255,cv.THRESH_OTSU) #binariser l'image : cv.THRESH_OTSU (il spécifie le seuil optimal)
                                                        #si un pixel a une valeur supérieure ou égale au seuil, il prendra la valeur 255 (blanc),
                                                        # et si sa valeur est inférieure, il prendra la valeur 0 (noir).
                                                        #cette fonction retourne deux valeurs le seuil et l'image binaire
Gaus = cv.GaussianBlur(ng_2012,(5,5),0)
Seuil2,bin_2012 = cv.threshold(Gaus,0,255,cv.THRESH_OTSU)

plt.figure()
plt.imshow(bin_2000,cmap="gray") #afficher l'image | colormap=gray
plt.title("2000")#titre de la figure
plt.show(block=False)#permet la continuation de l'execution du programme pendant l'affichage des figures

plt.figure()
plt.imshow(bin_2012,cmap="gray")
plt.title("2012")
plt.show(block=False)


kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(15,15)) #choisir l'element structurant 
                                                            #cv.MORPH_ELLIPSE:Dans notre cas on choisi un ellipse de taille 15x15
imFermeture_2000 = cv.morphologyEx(bin_2000, cv.MORPH_CLOSE, kernel)#Appliquer une fermeture(cv.MORPH_CLOSE) a notre image
                                                                    #en utilisant l'element structurant déjà choisi
                                                                                    
plt.figure()
plt.subplot(122)
plt.imshow(imFermeture_2000, cmap="gray")
plt.title('Image 2000 après fermeture')
plt.show(block=False)

plt.subplot(121)
plt.imshow(bin_2000, cmap="gray")
plt.title('Image 2000 binarisée')
plt.show(block=False)


imFermeture_2012 = cv.morphologyEx(bin_2012, cv.MORPH_CLOSE, kernel)
plt.figure()
plt.subplot(122)
plt.imshow(imFermeture_2012, cmap="gray")
plt.title('Image 2012 après fermeture')
plt.show(block=False)

plt.subplot(121)
plt.imshow(bin_2012, cmap="gray")
plt.title('Image 2012 binarisée')
plt.show(block=False)


# total des zones vides
total_couverture_2000 = np.sum(imFermeture_2000)/255 #calcul du nombre total des pixels ayant une couleur blanche
                                                    #puisque lors de la binarisation les pixel blancs avaient une intensité 255
                                                    #c'est pourquoi on divise par 255 
total_couverture_2012 = np.sum(imFermeture_2012)/255

print("zones vides en 2000 : {0:.2f}".format(total_couverture_2000/imFermeture_2000.size*100))#affichage du pourcentage
print("zones vides en 2012 : {0:.2f}".format(total_couverture_2012/imFermeture_2012.size*100))

#difference entre les deux images
diff = np.bitwise_xor(imFermeture_2000, imFermeture_2012)#il nous donne la difference entre les deux images
                                                        #blanc et blanc ->noir
                                                        #noir et noir ->noir
plt.figure()
plt.imshow(diff, cmap="gray")
plt.title('Différence entre 2000 et 2012 ')
plt.show(block=False)

total_couverture_diff= np.sum(diff)/255.
print("zones déforestées entre 2000 et 2012: {0:.2f}".format(total_couverture_diff/diff.size*100))

rgb_2000 = cv.imread('amazon_2000.png')
rgb_2012 = cv.imread('amazon_2012.png')

plt.figure()
plt.subplot(221)
plt.imshow(rgb_2000)
plt.title('Image 2000')
plt.show(block=False)

plt.subplot(222)
plt.imshow(rgb_2012)
plt.title('Image 2012')
plt.show(block=False)

for i in range(0,diff.shape[0]):
    for j in range(0,diff.shape[1]):
        if(diff[i][j]!=0):
            rgb_2012[i][j]=np.array([255,0,0])

plt.subplot(223)
plt.imshow(rgb_2012)
plt.title("zones déforestées entre 2000 et 2012:")
plt.show()