# Mengimpor Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Mengimpor dataset
dataku = pd.read_csv('harga_rumah.csv')

# Mendeteksi data kosong
kolom_na = [kolom for kolom in dataku.columns if dataku[kolom].isnull().sum() > 0]
var_na = dataku[kolom_na]

# Menghitung presentase var berisi NaN
var_na.isnull().mean()*100

# Visualisasi untuk variabel yang memiliki data kosong
def analisis_data_na(data, col):
    data = data.copy()
    # Cek setiap variabel jika ada NaN maka 1, jika tidak maka 0
    data[col] = np.where(data[col].isnull(), 1, 0)
    # Sekarang kita memiliki data binary 0 dan 1
    # Selanjutnya mengelompokkan terhadap SalePrice (var dependent)
    data.groupby(col)['SalePrice'].median().plot.bar()
    plt.title(col)
    plt.tight_layout()
    plt.show()
    
# Membuat looping untuk visualisasi variabel data kosong
batas = len(kolom_na)
i = 1
for kolom in kolom_na:
    i += 1
    analisis_data_na(dataku, kolom)
    if i <= batas: plt.figure()

# Menganalisis kolom-kolom numerik
# dtypes 'O' adalah object (string)
kolom_numerik = [kolom for kolom in dataku.columns if dataku[kolom].dtypes != 'O']
numerik = dataku[kolom_numerik]

'''
Variabel yang berhubungan dengan waktu:
    YearBuilt = kapan rumah dibangun
    YearRemodAdd = kapan rumah direnov
    GarageYrBlt = kapan garasinya dibangun
    YrSold = kapan rumahnya dijual
'''

# Kolom yang berhubungan dengan waktu (tahun)
kolom_tahun = [kolom for kolom in kolom_numerik if 'Year' in kolom or 'Yr' in kolom]
tahun = dataku[kolom_tahun]

# Visualisasi antara tahun penjualan dengan SalePrice
dataku.groupby('YrSold')['SalePrice'].median().plot()
plt.ylabel('Nilai Median Harga Jual Rumah')
plt.title('Perubahan Harga Rumah Pertahun')
plt.tight_layout()

# Analisis hubungan selisih antara variabel tahun (YearBuilt, YearRemodAdd, GarageYrBlt) dengan YrSold
def analisis_data_tahun(data, col):
    data = data.copy()
    # Melihat selisih antara kolom tahun dengan YrSold
    data[col] = data['YrSold'] - data[col]
    plt.scatter(data[col], data['SalePrice'])
    plt.ylabel('Harga Jual Rumah')
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()
    
# Looping untuk plotting data tahun
batas = len(kolom_tahun)
i = 1
for kolom in kolom_tahun:
    if kolom != 'YrSold':
        i += 1
        analisis_data_tahun(dataku, kolom)
        if i < batas: plt.figure()

# Menganalisis data diskrit (discrete)
kolom_diskrit = [kolom for kolom in kolom_numerik if len(dataku[kolom].unique()) <= 15 and kolom not in kolom_tahun + ['Id']]
diskrit = dataku[kolom_diskrit]

# Analisis data diskrit untuk plotting (hub antara data diskrit dengan SalePrice)
def analisis_data_diskrit(data, col):
    data = data.copy()
    data.groupby(col)['SalePrice'].median().plot.bar()
    plt.title(col)
    plt.ylabel('Median Harga Jual')
    plt.tight_layout()
    plt.show()
    
# Looping untuk plotting kolom diskrit
batas = len(kolom_diskrit)
i = 1
for kolom in kolom_diskrit:
    i += 1
    analisis_data_diskrit(dataku, kolom)
    if i <= batas: plt.figure()
    
# Menganalisis data kontinu (continue)
kolom_kontinu = [kolom for kolom in kolom_numerik if kolom not in kolom_diskrit + kolom_tahun + ['Id']]
kontinu = dataku[kolom_kontinu]

# Melihat len(var kontinu)
for i in kontinu:
    print(len(dataku[i].unique()))
    
# Fungsi untuk kolom_kontinu
def analisis_data_kontinu(data, col):
    data = data.copy()
    data[col].hist(bins=20)
    plt.ylabel('Jumlah Rumah')
    plt.xlabel(col)
    plt.title(col)
    plt.tight_layout()
    plt.show()
    
# Lopping untuk plotting kolom kontinu
batas = len(kolom_kontinu)
i = 1
for kolom in kolom_kontinu:
    i += 1
    analisis_data_kontinu(dataku, kolom)
    if i <= batas: plt.figure()

# Melakukan proses logtransform
def analisis_logtransform(data, col):
    data = data.copy()
    # Skala logaritmik tidak memperhitungkan 0 dan negatif, maka di skip
    if any(data[col] <= 0):
        pass
    else:
        data[col] = np.log(data[col]) # Proses Logtransform
        data[col].hist(bins=20)
        plt.ylabel('Jumlah Rumah')
        plt.xlabel(col)
        plt.title(col)
        plt.tight_layout()
        plt.show()
        
# Menentukan batas dan membuat variabel kolom_kontinu_log
batas = 0
kolom_kontinu_log = []
for kolom in kolom_kontinu:
    if any(dataku[kolom] <= 0):
        pass
    else:
        kolom_kontinu_log.append(kolom)
        batas += 1
kontinu_log = dataku[kolom_kontinu_log]

# Looping untuk plotting kolom_kontinu_log
i = 1
for kolom in kolom_kontinu_log:
    i += 1
    analisis_logtransform(dataku, kolom)
    if i <= batas: plt.figure()

# Analisis hubungan antara variabel kontinu_logtransform dengan SalePrice
def analisis_logtransform_scatter(data, col):
    data = data.copy()
    if any(data[col] <= 0):
        pass
    else:
        data[col] = np.log(data[col])
        # Proses logtransform SalePrice
        data['SalePrice'] = np.log(data['SalePrice'])
        # Plotting
        plt.scatter(data[col], data['SalePrice'])
        plt.ylabel('Harga Rumah')
        plt.xlabel(col)
        plt.tight_layout()
        plt.show()
        
# Looping untuk plotting kolom_kontinu_log
i = 1
for kolom in kolom_kontinu_log:
    if kolom != 'SalePrice':
        i += 1
        analisis_logtransform_scatter(dataku, kolom)
        if i < batas: plt.figure()
        
# Analisis outlier
def analisis_outlier(data, col):
    data = data.copy()
    if any(data[col] <= 0):
        pass
    else:
        data[col] = np.log(data[col])
        data.boxplot(column=col)
        plt.title(col)
        plt.ylabel(col)
        plt.tight_layout()
        plt.show()
        
# Looping untuk plotting outlier
i = 1
for kolom in kolom_kontinu_log:
    i += 1
    analisis_outlier(dataku, kolom)
    if i <= batas: plt.figure()
    
# Variabel kategori (nominal)
kolom_kategori = [kolom for kolom in dataku.columns if dataku[kolom].dtypes == 'O' ]
kategori = dataku[kolom_kategori]

# Mengecek item setiap kolom kategori
kategori.nunique()

# Pengecekan variabel kategori yang jarang 
def analisis_var_jarang(data, col, persentase):
    data = data.copy()
    isi = data.groupby(col)['SalePrice'].count() / len(data)
    return isi[isi < persentase]

# Looping untuk menghitung variabel jarang
for kolom in kolom_kategori:
    print(analisis_var_jarang(dataku, kolom, 0.01), '\n')

# Looping untuk plotting kolom_kategori
i = 1
batas = len(kolom_kategori)
for kolom in kolom_kategori:
    i += 1
    analisis_data_diskrit(dataku, kolom)
    if i <= batas: plt.figure()