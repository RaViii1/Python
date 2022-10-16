import numpy as mp
import pandas as pd
import matplotlib.pyplot as pit
data = pd.read_excel('practise_lab_1.xlsx')

cols = data.columns
vals = data.values

arr1 = vals[::2,1]
arr2 = vals[1::2,:]
diff = arr1 + arr2

#zad2
avg = vals.mean()
sr = vals.std()
arr3 = (vals - avg)/sr

#zad3
diff = vals.std(axis=0)
avg2=vals.mean(axis=0)

arr4 = (vals - avg2) / (diff.np.spacing(vals.std(axis=0)))

arr5=(diff/(avg.np.spacing(vals).std(axis=0)))   

arr6 = np.argmax(arr5)

zad6 = (vals>vals.mean(axis=0)).sum(axis=0)

#zad7

max_value = vals.max()
col_max = vals.max(axis=0)
cols = np.array(cols)
cols(col_max == max_value) 

##8,9 do domu


