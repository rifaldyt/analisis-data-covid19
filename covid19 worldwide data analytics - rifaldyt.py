# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

covid_df = pd.read_csv('covid_worldwide.csv')
covid_df.head(10)

covid_df.columns

dataframe = covid_df.rename(columns = {'Serial Number' : 'id', 
                                       'Country':'negara', 
                                       'Total Cases':'jumlah_kasus', 
                                       'Total Deaths':'jumlah_kematian', 
                                       'Total Recovered':'jumlah_kesembuhan', 
                                       'Active Cases':'kasus_aktif', 
                                       'Total Test':'jumlah_test', 
                                       'Population':'populasi'})
dataframe.head(10)

dataframe.dtypes

change_name_column = ['jumlah_kasus', 'jumlah_kematian', 'jumlah_kesembuhan', 'kasus_aktif', 'jumlah_test', 'populasi']

dataframe['jumlah_kasus'] = dataframe['jumlah_kasus'].str.replace(',', '').astype('float64')
dataframe['jumlah_kematian'] = dataframe['jumlah_kematian'].str.replace(',', '').astype('float64')
dataframe['jumlah_kesembuhan'] = dataframe['jumlah_kesembuhan'].str.replace(',', '').astype('float64')
dataframe['kasus_aktif'] = dataframe['kasus_aktif'].str.replace(',', '').astype('float64')
dataframe['jumlah_test'] = dataframe['jumlah_test'].str.replace(',', '').astype('float64')
dataframe['populasi'] = dataframe['populasi'].str.replace(',', '').astype('float64')


dataframe.dtypes

dataframe.isna().sum()

plt.figure(figsize=(10, 4))
ax = sns.heatmap(dataframe.isnull(), cbar=True, cmap="plasma", yticklabels=False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

plt.xlabel("Nama Kolom")
plt.title("Nilai yang Hilang pada Setiap Kolom")
plt.show()

dataframe[dataframe['jumlah_kesembuhan'].isna()]

"""A lot of these countries have missing values in more than one column. Therefore, I decide to remove these countries from the DataFrame."""

#Dropping Rows with Null Values in total_recovered columns

dataframe.dropna(subset=['jumlah_kesembuhan'], inplace=True)
dataframe.dropna(subset=['kasus_aktif'], inplace=True)

dataframe.tail()

dataframe[dataframe['populasi'].isna()]

"""dikutip oleh worldometer berdasarkan data UN. Populasi China per 27 April 2023 sebanyak 1,454,931,913, Kapal Diamond Princess 3,711, dan Zaandam 76,804. Tambahkan manua saja"""

dataframe.loc[dataframe['negara'] == 'China', 'populasi'] = 1_454_931_913

dataframe.info()

data_prep = dataframe[~dataframe.isnull().any(axis=1)]
data_prep.reindex()

data_prep.info()

# Jmlah kematian yang NaN isi dengan nilai 0
dataframe['jumlah_kematian'].fillna(0, inplace=True)
dataframe['jumlah_test'].fillna(0, inplace=True)

dataframe.drop(columns='id', inplace=True)
dataframe.reset_index(inplace=True)

dataframe.tail(10)

dataframe.drop([209, 206], inplace = True)

dataframe.tail()

"""Analysis"""

kasus_terbanyak = dataframe[['negara','jumlah_kasus']].sort_values(by='jumlah_kasus',ascending=False).head(15)

plt.figure(figsize=(12,7))
sns.set_style("whitegrid")
color = sns.light_palette("#79C")
explode = [0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0]

plt.title("15 Negara Terbanyak berdasarkan Jumlah Kasus Infeksi COVID-19")
plt.pie(kasus_terbanyak['jumlah_kasus'], labels=kasus_terbanyak['negara'], explode=explode, colors = color, autopct='%1.1f%%')

kematian_terbanyak = dataframe[['negara','jumlah_kematian']].sort_values(by='jumlah_kematian',ascending=False).head(15)

plt.figure(figsize=(12,7))
sns.set_style("whitegrid")
color = sns.cubehelix_palette(gamma=.5)
explode = [0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0]

plt.title("15 Negara Terbanyak berdasarkan Jumlah Kasus Kematian COVID-19")
plt.pie(kematian_terbanyak['jumlah_kematian'], labels=kematian_terbanyak['negara'], explode=explode, colors = color, autopct='%1.1f%%')

kematian_sedikit = dataframe[['negara','jumlah_kematian']].sort_values(by='jumlah_kematian',ascending=True).head(15)

plt.figure(figsize=(12,7))
sns.set_style("whitegrid")
color = sns.cubehelix_palette(gamma=.5)
explode = [0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0]

plt.title("15 Negara Paling Sedikit berdasarkan Jumlah Kasus Kematian COVID-19")
plt.pie(kematian_sedikit['jumlah_kematian'], labels = kematian_sedikit['negara'], explode = explode, colors = color, autopct='%1.1f%%')

kesembuhan_terbanyak = dataframe[['negara','jumlah_kesembuhan']].sort_values(by='jumlah_kesembuhan',ascending=False).head(15)

plt.figure(figsize=(12,7))
sns.set_style("whitegrid")
color = sns.light_palette("seagreen")
explode = [0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0]

plt.title("15 Negara Terbanyak berdasarkan Jumlah Kesembuhan COVID-19")
plt.pie(kesembuhan_terbanyak['jumlah_kesembuhan'], labels=kesembuhan_terbanyak['negara'], explode=explode, colors = color, autopct='%1.1f%%')

kasus_aktif_terbanyak = dataframe[['negara','kasus_aktif']].sort_values(by='kasus_aktif',ascending=False).head(15)

plt.figure(figsize=(12,7))
sns.set_style("whitegrid")
color = sns.cubehelix_palette(dark=.10, light=.90)
explode = [0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0]

plt.title("15 Negara Terbanyak berdasarkan Kasus Aktif COVID-19")
plt.pie(kasus_aktif_terbanyak['kasus_aktif'], labels=kasus_aktif_terbanyak['negara'], explode=explode, colors = color, autopct='%1.1f%%')

sns.set_style('darkgrid')
fig, ax = plt.subplots(figsize=(10, 6))

sns.lineplot(data=dataframe.head(15), x='negara', y='jumlah_kasus', label='jumlah_kasus', ax=ax)
sns.lineplot(data=dataframe.head(15), x='negara', y='jumlah_kematian', label='jumlah_kematian', ax=ax)
sns.lineplot(data=dataframe.head(15), x='negara', y='jumlah_kesembuhan', label='jumlah_kesembuhan', ax=ax)

ax.set_title('Grafik Jumlah Kasus, Jumlah Kematian, dan Jumlah Kesembuhan ')
ax.set_xlabel('Negara')
plt.xticks(rotation=90)
ax.set_ylabel('Presentase Kasus')
ax.legend()

plt.show()

sns.set_style('darkgrid')
sp_kasus = sns.scatterplot(data=dataframe, x='populasi', y='jumlah_kasus')

sp_kasus.set_xscale("log")
sp_kasus.set_yscale("log")

sp_kasus.set_xlabel('Populasi')
sp_kasus.set_ylabel('Jumlah Kasus')
sp_kasus.set_title('Korelasi antara Populasi dengan Jumlah Kasus')

plt.show()

sns.set_style('darkgrid')
sp_kasus_2 = sns.scatterplot(data=dataframe, x='jumlah_kematian', y='jumlah_kematian')

sp_kasus_2.set_xscale("log")
sp_kasus_2.set_yscale("log")

sp_kasus_2.set_xlabel('Jumlah Kasus')
sp_kasus_2.set_ylabel('Jumlah Kematian')
sp_kasus_2.set_title('Korelasi antara Jumlah Kasus dengan Jumlah Kematian')

plt.show()

sns.set_style('darkgrid')
sp_kasus_3 = sns.scatterplot(data=dataframe, x='jumlah_kasus', y='jumlah_kesembuhan')

sp_kasus_3.set_xscale("log")
sp_kasus_3.set_yscale("log")

sp_kasus_3.set_xlabel('Jumlah Kasus')
sp_kasus_3.set_ylabel('Jumlah Kesembuhan')
sp_kasus_3.set_title('Korelasi antara Jumlah Kasus dengan Jumlah Kesembuhan')

plt.show()

sns.set_style('darkgrid')
sp_kasus_4 = sns.scatterplot(data=dataframe, x='jumlah_kasus', y='kasus_aktif')

sp_kasus_4.set_xscale("log")
sp_kasus_4.set_yscale("log")

sp_kasus_4.set_xlabel('Jumlah Kasus')
sp_kasus_4.set_ylabel('Kasus Aktif')
sp_kasus_4.set_title('Korelasi antara Jumlah Kasus dengan Jumlah Kasus Aktif')

plt.show()

sns.set_style('darkgrid')
sp_kasus_5 = sns.scatterplot(data=dataframe, x='jumlah_kematian', y='jumlah_kesembuhan')

sp_kasus_5.set_xscale("log")
sp_kasus_5.set_yscale("log")

sp_kasus_5.set_xlabel('Jumlah Kematian')
sp_kasus_5.set_ylabel('Jumlah Kesembuhan')
sp_kasus_5.set_title('Korelasi antara Jumlah  Kematian dengan Jumlah Kesembuhan')

plt.show()

sns.set_style('darkgrid')
sp_kasus_6 = sns.scatterplot(data=dataframe, x='jumlah_kesembuhan', y='kasus_aktif')

sp_kasus_6.set_xscale("log")
sp_kasus_6.set_yscale("log")

sp_kasus_6.set_xlabel('Jumlah Kesembuhan')
sp_kasus_6.set_ylabel('Kasus Aktif')
sp_kasus_6.set_title('Korelasi antara Jumlah Kesembuhan dengan Jumlah Kasus Aktif')

plt.show()

sns.set_style('darkgrid')
sp_kasus_7 = sns.scatterplot(data=dataframe, x='kasus_aktif', y='jumlah_test')

sp_kasus_7.set_xscale("log")
sp_kasus_7.set_yscale("log")

sp_kasus_7.set_xlabel('Kasus Aktif')
sp_kasus_7.set_ylabel('Jumlah Test')
sp_kasus_7.set_title('Korelasi antara Jumlah Kasus Aktif dengan Jumlah Test')

plt.show()

sns.set_style('darkgrid')
sp_kasus_8 = sns.scatterplot(data=dataframe, x='populasi', y='jumlah_test')

sp_kasus_8.set_xscale("log")
sp_kasus_8.set_yscale("log")

sp_kasus_8.set_xlabel('Populasi')
sp_kasus_8.set_ylabel('Jumlah Test')
sp_kasus_8.set_title('Korelasi antara Populasi dengan Jumlah Test')

plt.show()

"""Map"""

pip install geopandas

import geopandas as gpd;

map = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
merged = map.merge(dataframe, left_on='name', right_on='negara')

#Map Jumlah Kasus Terbanyak
fig, ax = plt.subplots(figsize=(17, 13))
ax.set_title('Visualisasi Map Jumlah Kasus COVID-19')
merged.plot(column='jumlah_kasus', cmap='Blues', ax=ax, legend=True, legend_kwds={'label': "Presentase Jumlah Kasus",'orientation' : 'horizontal'})
plt.show()

"""Model Prediksi

Membuat model Regresi Linier untuk memprediksi jumlah total orang yang sembuh dari Covid per negara.
"""

dataframe['populasi_log'] = np.log(dataframe['populasi'])
dataframe['jumlah_kasus_log'] = np.log(dataframe['jumlah_kasus'])
dataframe['jumlah_kesembuhan_log'] = np.log(dataframe['jumlah_kesembuhan'])

#buat nilai x dan y

X = dataframe[['populasi_log', 'jumlah_kasus_log']]
y = dataframe['jumlah_kesembuhan_log']

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
y_pred

plt.scatter(y_test, y_pred)
plt.xlabel('Nilai Aktual')
plt.ylabel('Nilai Prediksi')
plt.title('Nilai Aktual vs Nilai Prediksi dalam Jumlah Kesembuhan')

plt.show()

