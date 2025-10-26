import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns # for data visualiztions

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
df = pd.read_csv(r"C:\Users\EjGam\Downloads\Data1002-A1-main\Notebooks\Set3\dataset1_cleaned.csv")


categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
data = pd.get_dummies(df, columns = categorical_cols)


X = data.drop("HeartDisease", axis=1)  
y = data["HeartDisease"]  


from sklearn.model_selection import train_test_split

# First split: Train (70%) and Temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=0)

# Second split: Validation (15%) and Test (15%) from Temp
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)


print('Training set shape: ', X_train.shape, y_train.shape)
print('Validation set shape:', X_val.shape, y_val.shape)
print('Testing set shape:   ', X_test.shape, y_test.shape)

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectKBest, f_classif
ft = SelectKBest(score_func=f_classif, k=10)  # Select top 10 features
ft.fit(X_train, y_train)

# Transform datasets to keep only selected features
X_train_selected = ft.transform(X_train)
X_val_selected = ft.transform(X_val)
X_test_selected = ft.transform(X_test)

print("Selected Features:", X_train.columns[ft.get_support()])

# -------------------------------
# Preprocessing: Standardization
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected.astype(float))
X_val_scaled = scaler.transform(X_val_selected.astype(float))
X_test_scaled = scaler.transform(X_test_selected.astype(float))

print("Preprocessing complete. Shapes after scaling:")
print("Train:", X_train_scaled.shape, "Validation:", X_val_scaled.shape, "Test:", X_test_scaled.shape)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

from sklearn import metrics

mean_acc = np.zeros(20)
for i in range(1,21):
    #Train Model and Predict  
    knn = KNeighborsClassifier(n_neighbors = i).fit(X_train,y_train)
    yhat= knn.predict(X_test)
    acc = metrics.accuracy_score(y_test, yhat)
    mean_acc[i-1] = acc

    print(f"k={i}, Model: {knn}, Accuracy: {acc:.4f}")


print("\nMean Accuracy for k=1 to 20:", mean_acc)
best_k = mean_acc.argmax() + 1
print(f"Best k: {best_k} with accuracy: {mean_acc[best_k-1]:.4f}")

loc = np.arange(1,21,step=1.0)
plt.figure(figsize = (10, 6))
plt.plot(range(1,21), mean_acc)
plt.xticks(loc)
plt.xlabel('Number of Neighbors ')
plt.ylabel('Accuracy')
plt.show()

from sklearn.model_selection import GridSearchCV

grid_params = { 'n_neighbors' : [5,7,9,11,13,15],
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']}

gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose = 1, cv=3, n_jobs = -1)
g_res = gs.fit(X_train, y_train)

print(f"Best Score: {g_res.best_score_:.4f}")
print(f"Best Parameters: {g_res.best_params_}")

knn = KNeighborsClassifier(n_neighbors = 15, weights = 'distance', algorithm = 'brute',metric = 'manhattan')
knn.fit(X_train, y_train)

y_hat = knn.predict(X_train)
y_knn = knn.predict(X_test)

print('Training set accuracy: ', metrics.accuracy_score(y_train, y_hat))
print('Test set accuracy: ',metrics.accuracy_score(y_test, y_knn))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_knn))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_knn))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, X, y, cv =5)

print('Model accuracy: ',np.mean(scores))

from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, y_knn)
recall = recall_score(y_test, y_knn)
f1 = f1_score(y_test, y_knn)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")