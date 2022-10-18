import numpy as np
import pandas as pd
##import matplotlib.pyplot as pit
data = pd.read_excel('F:/Python/practice_lab_1.xlsx')

cols = data.columns #Nazwy kolumn
vals = data.values  #wartosci w kolumnach

arr1 = vals[::2,1] #wszystkie kolumny, parzyste wiersze
arr2 = vals[1::2,:]#wszysykie kolumny, nieparzyste wiersze bo zaczynamy od 1
#diff = arr1 - arr2

#zad2
avg = vals.mean() #srednia dla calej tablicy
sr = vals.std()     #odychlenie std dla calej tablicy
arr3 = (vals - avg) / sr

#zad3
diff = vals.std(axis=0)
avg2=vals.mean(axis=0)
arr4 = (vals - avg2) / (diff.np.spacing(vals.std(axis=0)))

#zad4
arr5=(diff/(avg.np.spacing(vals).std(axis=0)))   

#zad5
arr6 = np.argmax(arr5)

#zad6 
zad6 = (vals>vals.mean(axis=0)).sum(axis=0)

#zad7
max_value = vals.max()
col_max = vals.max(axis=0)
cols = np.array(cols)
cols(col_max == max_value) 

#zad8
mask = vals == 0 #tworzenie maski dla wartosci = 0
ArrayForZeros = np.sum(mask, axis=0) # Sumowanie wystąpień 0 w poszczegolnych kolumnach tworzy nowa tablice
ZeroMax = max(ArrayForZeros) #Najwyzsza suma 0 w tablicy z sumą zer
name = pd.DataFrame(cols[ZeroMax == ArrayForZeros]) # porownojac nową tablicę  oraz max 0 do col otrzymujemy nazwe kolumny

#zad9
parzyste = vals[::2,:] #sumowanie zaczynajac od 0 skok co 2
nieparzyste = vals[1::2,:]#sumowanie zaczynajac od 1 skok co 2

parzyste_sum = np.sum(parzyste, axis=0) #Suma parzystych wartosci w kolumnach
nieparzyste_sum = np.sum(nieparzyste, axis=0)#Suma nieparzystych wartosci w kolumnach

mask = parzyste_sum > nieparzyste_sum #maska pokazujaca kolumny w ktorych parz>nieparz

tabNazw = np.array(cols)[mask] #nowa tablica
print(tabNazw)

