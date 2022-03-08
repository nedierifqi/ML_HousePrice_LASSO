# Mengimpor library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Library untuk membuat model
from sklearn.linear_model import Lasso

# Library untuk mengevaluasi model regresi
from sklearn.metrics import mean_squared_error, r2_score

# Library untuk membuat model yang kita buat disini dan di deploy hasilnya sama
import joblib

# Mengimpor dataset
X_train = pd.read_csv('xtrain.csv')
X_test = pd.read_csv('xtest.csv')

# Menentukan variabel dependen
y_train = X_train['SalePrice']
y_test = X_test['SalePrice']

# Mengimpor fitur terpilih
fitur = pd.read_csv('fitur_pilihan.csv')
fitur = fitur['0'].tolist() # Merubah df menjadi list
#fitur = fitur + ['LotFrontage']

# Mengurangi kolom training dan test set sesuai fitur
X_train = X_train[fitur]
X_test = X_test[fitur]

# Menyiapkan model
model_linear = Lasso(alpha=0.005, random_state=10)

# Training model
model_linear.fit(X_train, y_train)

# Kita jalankan joblib untuk proses selanjutnya
joblib.dump(model_linear, 'regresi_lasso.pkl')

# Memprediksi model
pred_train = model_linear.predict(X_train)
pred_test = model_linear.predict(X_test)

# Menghitung mse (mean squared error) dan rsme (root mean squared error)
# np.exp --> menghitung nilai e^x, karena skala sebelumnya adalah logaritmik
print('Train MSE: {:.2f}'.format(mean_squared_error(np.exp(y_train), np.exp(pred_train))))
print('Train RMSE: {:.2f}'.format(mean_squared_error(np.exp(y_train), np.exp(pred_train), squared=False)))
print('Train R^2: {:.2f}'.format(r2_score(np.exp(y_train), np.exp(pred_train))))

print()

print('Test MSE: {:.2f}'.format(mean_squared_error(np.exp(y_test), np.exp(pred_test))))
print('Test RMSE: {:.2f}'.format(mean_squared_error(np.exp(y_test), np.exp(pred_test), squared=False)))
print('Test R^2: {:.2f}'.format(r2_score(np.exp(y_test), np.exp(pred_test))))

'''
Jika MSE atau RMSE antara training dan test set tidak berbeda jauh, maka tidak terjadi overfitting
'''

print('Rataan Harga Rumah Test Set: ', int(np.exp(y_test.mean())))
print('Rataan Harga Rumah Prediksi Test Set: ', int(np.exp(pred_test.mean())))

# Evaluasi visual
plt.scatter(y_test, pred_test)
plt.xlabel('Harga Jual Rumah Sesungguhnya')
plt.ylabel('Harga Jual Rumah Prediksi')
plt.title('Evaluasi Hasil Prediksi')
plt.tight_layout()

# Evaluasi distribusi dari error yang dihasilkan
errors = y_test - pred_test
errors.hist(bins=30)

# Melihat tingkat utilitas setiap fitur
utilitas = pd.Series(np.abs(model_linear.coef_.ravel()))
utilitas.index = fitur
utilitas.sort_values(inplace=True, ascending=False)
utilitas.plot.bar()
plt.ylabel('Koefisien Lasso')
plt.xlabel('Tingkat Utilitas Fitur')
plt.tight_layout()