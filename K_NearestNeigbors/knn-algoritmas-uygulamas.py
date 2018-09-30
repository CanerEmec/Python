
# coding: utf-8

# # GİRİŞ
# 
# Bu çalışmada "Biomechanical features of orthopedic patients" veriseti kullanılarak  KNN Algoritması yazılmış ve yazılan bu algoritma sklearn kütüphanesindeki KNN sınıflandırıcı ile karşılaştırılmıştır.
# 
# Sırasıyla şu adımlar izlenmiştir;
# 
#    * Kütüphanelerin İmport Edilmesi.
#    * Verisetinin Alınması.
#    * Verisetinin İncelenmesi.
#    * Veri İçerisinden Özellikler İle Etiketlerin Çıkartılması Ve Etiketlerin İnteger Değer Olarak Belirtilmesi.
#    * Normalizasyon İşlemi.
#    * Verinin Eğitim Ve Test Verisi Olarak Ayrılması.
#    * KNN Algoritmasının Yazılması.
#       * Fonksiyonların Yazılması.
#       * Yazılan Algoritmanın Denenmesi.
#    * Sklearn İle KNN Algoritmasının Kodlanması.
#    * Algoritmaların Farklı k Değerleri İçin Karşılaştırılması.

# %%ADIM 1: Kütüphanelerin İmport Edilmesi.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import os



# %%ADIM 2: Verisetinin Alınması.

data = pd.read_csv('column_3C_weka.csv')


# %%ADIM 3: Verisetinin İncelenmesi.

# Verisetinin içeriğinden küçük bir kısım aşağıda görülmektedir.
data.head()


# %% ADIM 4: Veri İçerisinden Özellikler İle Etiketlerin Çıkartılması Ve Etiketlerin İnteger Değer Olarak Belirtilmesi.

# Veri sayısal olarak ifade ediliyor.

data.loc[:,'class'] = [1 if each == 'Normal' else 0 for each in data.loc[:,'class'] ]
Labels = data.loc[:,'class']

x = data.drop(["class"],axis = 1)


# %%ADIM 5: Normalizasyon İşlemi.

x_norm = (x - np.min(x))/(np.max(x) - np.min(x))

print("NORMALİZASYON İŞLEMİ ÖNCESİ:",
      "\nMin :")
print(np.min(x))
print("\nMax :")
print(np.max(x))


print("\n\nNORMALİZASYON İŞLEMİ SONRASI:",
      "\nMin :")
print(np.min(x_norm))
print("\nMax :")
print(np.max(x_norm))

#%% ADIM 6: Verinin Eğitim ve Test Verisi Olarak Ayrılması.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_norm, Labels, test_size = 0.3, random_state = 1)


# %%ADIM 7: KNN Algoritmasının Yazılması. 
# 
# K-Nearest Neighbors (K-En Yakın Komşu) algoritması şu şekilde işler;
# * Test noktası seçilir.
# * Diğer tüm noktalar için test noktasına olan uzaklıklar hesaplanır.
# * En yakın "K" tane nokta bulunur.
# * Bu "K" tane noktanın etiketine bakılarak test noktası sınıflandırılır. Örneğin yukarıdaki resimde k=3'tür ve bu 3 komşunun 2 tanesi ikinci sınıfa, 1 tanesi birici sınıfa aittir. Dolayısıyla test noktası için "ikinci sınıfa aittir" diyebiliriz.

# %%ADIM 7.1: Fonksiyonların Yazılması


# Buradaki fonksiyon 2 nokta arasındaki uzaklığı hesaplamaktadır.
# Distance = Sqrt(Sum((p1-p2)^2)) 
def Distance(point_1,point_2):
    total = 0
    for idx in range(len(point_1)):
        total = total + (point_1[idx] - point_2[idx])**2
    return total**0.5
    
    

def K_NNeighbors(k_value, x_train, y_train, x_test):
    y_predict = []
    
    #Herbir test noktası için diğer tüm noktalara olan uzaklıklar hesaplanıyor.
    #Bulunan uzaklıklar etiketlerle beraber 'Neighbors' değişkeninde tutuluyor.
    for idx_test in range(x_test.shape[0]):
        Neighbors = []
        test_point = x_test[idx_test]
        for idx_rows in range(x_train.shape[0]):
            train_point = x_train[idx_rows]
            Neighbors.append([Distance(test_point, train_point),y_train[idx_rows]])
        
        # Her bir komşunun test noktasına olan uzaklığı bulunuyor.En yakın 'K' tane komşuyu seçmek için 
        # öncelikle komşular uzaklıklarına göre küçükten büyüğe doğru sıralanıyor..
        # Daha sonra k tane komşu seçilip içerisinden etiket(label) değerleri çekiliyor.
        Neighbors.sort()
        Neighbors = Neighbors[0:k_value]
        Labels = [n[1] for n in Neighbors]
        
        # En yakın k tane komşunun sahip olduğu etiketlerin frekansları bulunuyor ve en yüksek frekansa sahip
        # etiket test noktasını sınıflamakta kullanılıyor.
        from itertools import groupby
        Freq = [[len(list(group)), key] for key, group in groupby(Labels)]
        y_predict.append(max(Freq)[1])
    return y_predict
        
            
    
    


#%% ADIM 7.2: Yazılan Algoritmanın Denenmesi
y_predicted = K_NNeighbors(5, np.array(x_train), np.array(y_train), np.array(x_test))


# Yazılan algoritmanın doğruluğu ölçülüyor.
from sklearn.metrics import accuracy_score
print("Accuracy of (my)KNN algorithm: ", accuracy_score(y_test, y_predicted))


#%% ADIM 8: Sklearn İle KNN Algoritmasının Kodlanması


# KNN Modeli
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)

print("Accuracy of KNN(sklearn) algorithm: ",knn.score(x_test, y_test))


#%% ADIM 9: Algoritmaların Farklı k Değerleri İçin Karşılaştırılması.

# Algoritmaların Karşılaştırılması.
score_list_sklearn = []
score_list_myknn = []

for each in range(1,50):
    sklearn_knn = KNeighborsClassifier(n_neighbors=each)
    sklearn_knn.fit(x_train, y_train)
    
    y_predicted = K_NNeighbors(each, np.array(x_train), np.array(y_train), np.array(x_test))
    
    score_list_myknn.append(accuracy_score(y_test, y_predicted))
    score_list_sklearn.append(sklearn_knn.score(x_test,y_test))
 
plt.plot(range(1, 50), score_list_sklearn)
plt.xlabel("k value")
plt.ylabel("Accuracy")
plt.title("KNN With Sklearn")
plt.show()

plt.plot(range(1, 50), score_list_myknn)
plt.xlabel("k value")
plt.ylabel("Accuracy")
plt.title("My KNN")
plt.show()

