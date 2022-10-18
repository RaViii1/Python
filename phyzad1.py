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
