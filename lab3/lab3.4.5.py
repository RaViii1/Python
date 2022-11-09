# %% zad 3.4

features = data.columns
vals = data.values.astype(np.float)
y = data['Loan_Status'].astype(np.float)
X = data.drop(columns = ['Loan_Status']).values.astype(np.float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# uniform  każdy taksamo distance - im dlaej tym gorzej ,

models = [kNN(weights='distance'), SVM(kernel= 'sigmoid')]
for model in models:
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    
#
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import SVC
models = [kNN(), SVC()]
for model in models:
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

#

features = data.columns
vals = data.values.astype(np.float)
y = data['Loan_Status'].astype(np.float)
X = data.drop(columns = ['Loan_Status']).values.astype(np.float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler_MinMaxScaler.transform(X_train)
X_test = sscaler_MinMaxScaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import SVC
models = [kNN(), SVC()]
for model in models:
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
