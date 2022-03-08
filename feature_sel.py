# Mengimpor Library
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

# Mengimpor dataset
X_train = pd.read_csv('xtrain.csv')
X_test = pd.read_csv('xtest.csv')

# Menyiapkan variabel dependen
y_train = X_train['SalePrice']
y_test = X_test['SalePrice']

# Membuang kolom yang tidak diperlukan
X_train.drop(['Id', 'SalePrice'], axis=1, inplace=True)
X_test.drop(['Id', 'SalePrice'], axis=1, inplace=True)

# Melakukan proses feature selection
pilih = SelectFromModel(Lasso(alpha=0.005, random_state=10))
# Kita menggunakan Lasso --> Least Absolute Shrinkage and Selection Operator
# Lasso digunakan di regresi untuk memilih feature dan regularisasi untuk meningkatkan kualitas prediksi dan akurasi
# SelectFromModel akan memilih semua kolom yang nilai coef_ dari hasil Lasso dengan nilai 1
pilih.fit(X_train, y_train)

# Melihat kolom (variabel) mana saja yang dipilih (True)
pilih.get_support()

# Membuat list kolom apa saja yang dipilih
pilihan_kolom = X_train.columns[(pilih.get_support())]

# Mencetak beberapa ringkasan
print('Jumlah kolom awal: {}'.format((X_train.shape[1])))
print('Jumlah kolom terpilih: {}'.format(len(pilihan_kolom)))
print('Jumlah kolom yang tidak terpilih: {}'.format(np.sum(pilih.estimator_.coef_ == 0)))

# Meng-export fitur pilihan ke dalam format csv
pd.Series(pilihan_kolom).to_csv('fitur_pilihan.csv', index=False)

coba = pd.read_csv('fitur_pilihan.csv')