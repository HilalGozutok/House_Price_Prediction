#############################################
# HOUSE PRICES PREDICTION PROJECT
#############################################
"""""
Veri kümesindeki her özelliğin kısaca açıklaması söyledir:

SalePrice - mülkün dolar cinsinden satış fiyatı. Bu, tahmin etmeye çalışılan hedef değişkendir.
MSSubClass: İnşaat sınıfı
MSZoning: Genel imar sınıflandırması
LotFrontage: Mülkiyetin cadde ile doğrudan bağlantısının olup olmaması
LotArea: Parsel büyüklüğü
Street: Yol erişiminin tipi
Alley: Sokak girişi tipi
LotShape: Mülkün genel şekli
LandContour: Mülkün düzlüğü
Utulities: Mevcut hizmetlerin türü
LotConfig: Parsel yapılandırması
LandSlope: Mülkün eğimi
Neighborhood: Ames şehir sınırları içindeki fiziksel konumu
Condition1: Ana yol veya tren yoluna yakınlık
Condition2: Ana yola veya demiryoluna yakınlık (eğer ikinci bir yer varsa)
BldgType: Konut tipi
HouseStyle: Konut sitili
OverallQual: Genel malzeme ve bitiş kalitesi
OverallCond: Genel durum değerlendirmesi
YearBuilt: Orijinal yapım tarihi
YearRemodAdd: Yeniden düzenleme tarihi
RoofStyle: Çatı tipi
RoofMatl: Çatı malzemesi
Exterior1st: Evdeki dış kaplama
Exterior2nd: Evdeki dış kaplama (birden fazla malzeme varsa)
MasVnrType: Duvar kaplama türü
MasVnrArea: Kare ayaklı duvar kaplama alanı
ExterQual: Dış malzeme kalitesi
ExterCond: Malzemenin dışta mevcut durumu
Foundation: Vakıf tipi
BsmtQual: Bodrumun yüksekliği
BsmtCond: Bodrum katının genel durumu
BsmtExposure: Yürüyüş veya bahçe katı bodrum duvarları
BsmtFinType1: Bodrum bitmiş alanının kalitesi
BsmtFinSF1: Tip 1 bitmiş alanın metre karesi
BsmtFinType2: İkinci bitmiş alanın kalitesi (varsa)
BsmtFinSF2: Tip 2 bitmiş alanın metre karesi
BsmtUnfSF: Bodrumun bitmemiş alanın metre karesi
TotalBsmtSF: Bodrum alanının toplam metre karesi
Heating: Isıtma tipi
HeatingQC: Isıtma kalitesi ve durumu
CentralAir: Merkezi klima
Electrical: elektrik sistemi
1stFlrSF: Birinci Kat metre kare alanı
2ndFlrSF: İkinci kat metre kare alanı
LowQualFinSF: Düşük kaliteli bitmiş alanlar (tüm katlar)
GrLivArea: Üstü (zemin) oturma alanı metre karesi
BsmtFullBath: Bodrum katındaki tam banyolar
BsmtHalfBath: Bodrum katındaki yarım banyolar
FullBath: Üst katlardaki tam banyolar
HalfBath: Üst katlardaki yarım banyolar
BedroomAbvGr: Bodrum seviyesinin üstünde yatak odası sayısı
KitchenAbvGr: Bodrum seviyesinin üstünde mutfak Sayısı
KitchenQual: Mutfak kalitesi
TotRmsAbvGrd: Üst katlardaki toplam oda (banyo içermez)
Functional: Ev işlevselliği değerlendirmesi
Fireplaces: Şömineler
FireplaceQu: Şömine kalitesi
Garage Türü: Garaj yeri
GarageYrBlt: Garajın yapım yılı
GarageFinish: Garajın iç yüzeyi
GarageCars: Araç kapasitesi
GarageArea: Garajın alanı
GarageQual: Garaj kalitesi
GarageCond: Garaj durumu
PavedDrive: Garajla yol arasındaki yol
WoodDeckSF: Ayaklı ahşap güverte alanı
OpenPorchSF: Kapı önündeki açık veranda alanı
EnclosedPorch: Kapı önündeki kapalı veranda alan
3SsPorch: Üç mevsim veranda alanı
ScreenPorch: Veranda örtü alanı
PoolArea: Havuzun metre kare alanı
PoolQC: Havuz kalitesi
Fence: Çit kalitesi
MiscFeature: Diğer kategorilerde bulunmayan özellikler
MiscVal: Çeşitli özelliklerin değeri
MoSold: Satıldığı ay
YrSold: Satıldığı yıl
SaleType: Satış Türü
SaleCondition: Satış Durumu
"""""
###########################
# IMPORT LIBRARIES & MODULES
###########################

import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pickle
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import warnings
from sklearn.exceptions import ConvergenceWarning
from helpers.data_prep import *
from helpers.eda import *
from helpers.helpers import *


###########################
# LOAD DATA
###########################

# train ve test setleri birleştirildi.
train = pd.read_csv(r"C:\Users\LENOVO\PycharmProjects\DSMLBC4\HAFTA_09\train.csv")
test = pd.read_csv(r"C:\Users\LENOVO\PycharmProjects\DSMLBC4\HAFTA_09\test.csv")
df = train.append(test).reset_index(drop=True)
df.head()

######################################
# EDA
######################################

check_df(df)

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df, car_th=10)

"""""
Observations: 2919
Variables: 81
cat_cols: 50
num_cols: 28
cat_but_car: 3
num_but_cat: 10
"""""
######################################
# KATEGORIK DEGISKEN ANALIZI
######################################

for col in cat_cols:
    cat_summary(df, col)

for col in cat_but_car:
    cat_summary(df, col)

for col in num_but_cat:
    cat_summary(df, col)

######################################
# SAYISAL DEGISKEN ANALIZI
######################################

df[num_cols].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]).T

for col in num_cols:
    num_summary(df, col, plot=True)


######################################
# TARGET ANALIZI
######################################

df["SalePrice"].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99])

######################
# KORELASYON ANALIZI
######################

# target ile bagımsız degiskenlerin korelasyonları
low_corrs, high_corrs = find_correlation(df, num_cols)

# Tüm değişkenler arasındaki korelasyon
import matplotlib.pyplot as plt
corr_matrix = df.corr()
sns.clustermap(corr_matrix, annot = True, fmt = ".2f", cmap = "viridis", figsize=(11,11))
plt.title("Correlation Between Features")
plt.show()

# Detaylı inceleme

threshold = 0.60
filtre = np.abs(corr_matrix["SalePrice"]) > threshold
corr_features=corr_matrix.columns[filtre].tolist()
sns.clustermap (df[corr_features].corr(), annot = True, fmt = ".2f",cmap = "viridis")
plt.title("Correlation Between Features w/ Corr Threshold 0.75")
plt.show()

######################################
# MISSING_VALUES
######################################
missing_values_table(df)
none_cols = ['GarageType','GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
             'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']

zero_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath','BsmtHalfBath',
             'GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea']

freq_cols = ['Exterior1st', 'Exterior2nd', 'KitchenQual',]

for col in zero_cols:
    df[col].replace(np.nan, 0, inplace=True)

for col in none_cols:
    df[col].replace(np.nan, "None", inplace=True)

for col in freq_cols:
    df[col].replace(np.nan, df[col].mode()[0], inplace=True)

df['MSZoning'] = df.groupby('MSSubClass')['MSZoning'].apply(lambda x: x.fillna(x.mode()[0]))
df['LotFrontage'] = df.groupby(['Neighborhood'])['LotFrontage'].apply(lambda x: x.fillna(x.median()))


######################################
# OUTLIERS
######################################

df["SalePrice"].describe().T

replace_with_thresholds(df, "SalePrice")

df["SalePrice"].describe().T

######################################
# RARE ENCODING
######################################

rare_analyser(df, "SalePrice", 0.01)
df = rare_encoder(df, 0.01)


######################################
# FEATURE ENGINEERING
######################################

# MSSubClass = İnşaat sınıfı
df["MSSubClass"] = df["MSSubClass"].astype(str)
# YrSold = Evin satıldığı yıl
df["YrSold"] = df["YrSold"].astype(str)
# MoSold = Evin satıldığı ay
df["MoSold"] = df["MoSold"].astype(str)

# MSZoning : Genel imar sınıflandırılması
df["MSZoning"].value_counts()
df.loc[(df["MSZoning"] == "RH"),"MSZoning"] = "RM"

# LotShape : Mülkün genel şekli
df["LotShape"].value_counts()
df.loc[(df["LotShape"] == "IR2"),"LotShape"] = "IR1"
df.loc[(df["LotShape"] == "IR3"),"LotShape"] = "IR1"

# LotConfig : Parsel yapılandırılması
df["LotConfig"].value_counts()
df.loc[(df["LotConfig"] == "Corner"),"LotConfig"] = "FR2"
df.loc[(df["LotConfig"] == "Inside"),"LotConfig"] = "FR2"
df.loc[(df["LotConfig"] == "CulDSac"),"LotConfig"] = "FR3"

# LandSlope : Mülkün eğimi
df.loc[(df["LandSlope"] == "Mod"),"LandSlope"] = "Sev"
df["LandSlope"].value_counts()

# Condition1 : Anayol veya tren yoluna yakınlık
df.loc[(df["Condition1"] == "Feedr"),"Condition1"] = "Artery"
df.loc[(df["Condition1"] == "RRAe"),"Condition1"] = "Artery"
df.loc[(df["Condition1"] == "RRAn"),"Condition1"] = "Norm"
df.loc[(df["Condition1"] == "PosN"),"Condition1"] = "PosA"
df.loc[(df["Condition1"] == "RRNe"), "Condition1"] = "PosA"
df.loc[(df["Condition1"] == "RRNn"), "Condition1"] = "PosA"

# HouseStyle : Konut stili
df.loc[(df["HouseStyle"] == "1.5Fin"),"HouseStyle"] = "1.5Unf"
df.loc[(df["HouseStyle"] == "2.5Unf"), "HouseStyle"] = "1.5Unf"
df.loc[(df["HouseStyle"] == "SFoyer"), "HouseStyle"] = "1.5Unf"
df.loc[(df["HouseStyle"] == "SLvl"), "HouseStyle"] = "1Story"
df.loc[(df["HouseStyle"] == "2.5Fin"), "HouseStyle"] = "2Story"


# Kontrol

df.loc[(df["MasVnrType"] == "BrkCmn"), "MasVnrType"] = "None"


df.loc[(df["GarageType"] == "2Types"), "GarageType"] = "Attchd"
df.loc[(df["GarageType"] == "Basment"), "GarageType"] = "Attchd"

df.loc[(df["GarageType"] == "2Types"), "GarageType"] = "Attchd"
df.loc[(df["GarageType"] == "CarPort"), "GarageType"] = "Detchd"


# Derecelendirme içeren değişkenler ordinal yapıya getirildi.

# ExterQual : Dış malzeme kalitesi
ext_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
df['ExterQual'] = df['ExterQual'].map(ext_map).astype('int')

# ExterCond : Malzemenin dışta mevcut durumu
ext_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
df['ExterCond'] = df['ExterCond'].map(ext_map).astype('Int64')

# BsmtQual: Bodrumun yüksekliği
bsm_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
df['BsmtQual'] = df['BsmtQual'].map(bsm_map).astype('Int64')
df['BsmtCond'] = df['BsmtCond'].map(bsm_map).astype('Int64')

# BsmtFinType1: Bodrum bitmiş alanının kalitesi
bsmf_map = {'None': 0,'Unf': 1,'LwQ': 2,'Rec': 3,'BLQ': 4,'ALQ': 5,'GLQ': 6}
df['BsmtFinType1'] = df['BsmtFinType1'].map(bsmf_map).astype('Int64')
df['BsmtFinType2'] = df['BsmtFinType2'].map(bsmf_map).astype('Int64')

heat_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
# HeatingQC: Isıtma kalitesi ve durumu
df['HeatingQC'] = df['HeatingQC'].map(heat_map).astype('Int64')

# KitchenQual: Mutfak kalitesi
df['KitchenQual'] = df['KitchenQual'].map(heat_map).astype('Int64')

# GarageCond: Garaj durumu
df['GarageCond'] = df['GarageCond'].map(bsm_map).astype('Int64')

# GarageQual: Garaj kalitesi
df['GarageQual'] = df['GarageQual'].map(bsm_map).astype('Int64')

# NEW_TOTAL_BATH : Evdeki toplam banyo sayısı ( Bodrum katındaki tam banyolar + Üst katlardaki tam banyolar + (Üst katlardaki yarım banyolar) * 0.5)
# df["NEW_TOTAL_BATH"] = df['BsmtFullBath'] + df['BsmtHalfBath'] * 0.5 + df['FullBath'] + df['HalfBath']
df["NEW_TOTAL_BATH"] = df['BsmtFullBath'] + df['FullBath'] + df['HalfBath'] * 0.5

# Toplam kat sayısı
df['NEW_TotalSF'] = (df["BsmtFinSF1"] + df["BsmtFinSF2"] + df["1stFlrSF"] + df["2ndFlrSF"])

# 1. kat ve bodrum m^2
df["NEW_SF"] = df["1stFlrSF"] + df["TotalBsmtSF"]

# toplam m^2
df["NEW_TOTAL_M^2"] = df["NEW_SF"] + df["2ndFlrSF"]

# Garaj alanı ve m^2'lerin toplamı
df["NEW_SF_G"] = df["NEW_SF"] + df["GarageArea"]

# Toplam veranda alanı
df["NEW_TotalPorchSF"] = (df["OpenPorchSF"] + df["3SsnPorch"] + df["EnclosedPorch"] + df["ScreenPorch"] + df['WoodDeckSF'])

# Evin yaşı
df["NEW_BOMBA"] = df["YearRemodAdd"] + df["YearBuilt"]

# Evin yaşını kategoriye ayırdık
df["NEW_BOMBA_CAT"] = pd.qcut(df['NEW_BOMBA'], 5, labels=[1, 2, 3, 4, 5])

# Evin kalitesiyle ilgili değişkenler
df["NEW_QUAL_COND"] = df['OverallQual'] + df['OverallCond']
df["NEW_BSMT_QUAL_COND"] = df['BsmtQual'] + df['BsmtCond']
df["NEW_EX_QUAL_COND"] = df['ExterQual'] + df['ExterCond']
df["NEW_BSMT_QUAL_COND"] = df['GarageQual'] + df['BsmtCond']

# İyi durumdaki evlere FLAG ataması
df['NEW_BEST'] = (df['NEW_QUAL_COND'] >= 14).astype('Int64')


# Havuzlu evler
df["NEW_HAS_POOL"] = (df['PoolArea'] > 0).astype('Int64')


# Lux evler
df.loc[(df['Fireplaces'] > 0) & (df['GarageCars'] >= 3), "NEW_LUX"] = 1
df["NEW_LUX"].fillna(0, inplace=True)
df["NEW_LUX"] = df["NEW_LUX"].astype(int)

# Toplam alan
df["NEW_AREA"] = df["GrLivArea"] + df["GarageArea"]
# df.groupby("MiscVal").agg({"SalePrice": ["count", "mean","median"]})

df.loc[(df['TotRmsAbvGrd'] >= 7) & (df['GrLivArea'] >= 1800), "NEW_TOTAL_GR"] = 1

# m^2/oda
# df["NEW_TOTAL_ROOM"] = df["BedroomAbvGr"] + df["KitchenAbvGr"] + df["TotRmsAbvGrd"]
df["NEW_ROOM_AREA"] = df["NEW_TOTAL_M^2"] / df["TotRmsAbvGrd"]


# Create Cluster
ngb = df.groupby("Neighborhood").SalePrice.mean().reset_index()
ngb["NEW_CLUSTER_NEIGHBORHOOD"] = pd.cut(df.groupby("Neighborhood").SalePrice.mean().values, 4, labels=range(1, 5))
df = pd.merge(df, ngb.drop(["SalePrice"], axis=1), how="left", on="Neighborhood")

# Garaj yaşı
df["NEW_GARAGEBLTAGE"] = df.GarageYrBlt - df.YearBuilt

######################################
# RARE ENCODING
######################################

rare_analyser(df, "SalePrice", 0.01)
df = rare_encoder(df, 0.01)

drop_list = ["Street", "SaleCondition", "Functional", "Condition2", "Utilities", "SaleType", "MiscVal",
             "Alley", "LandSlope", "PoolQC", "MiscFeature", "Electrical", "Fence", "RoofStyle", "RoofMatl",
             "FireplaceQu"]


cat_cols = [col for col in cat_cols if col not in drop_list]

for col in drop_list:
    df.drop(col, axis=1, inplace=True)

######################################
# ONE-HOT ENCODING
######################################

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df, car_th=10)
cat_cols = cat_cols + cat_but_car
df = one_hot_encoder(df, cat_cols, drop_first=True)

"""""
Observations: 2919
Variables: 84
cat_cols: 43
num_cols: 36
cat_but_car: 5
num_but_cat: 27
"""""

######################################
# TRAIN TEST'IN AYRILMASI
######################################

train_df = df[df['SalePrice'].notnull()]
test_df = df[df["SalePrice"].isnull()].drop("SalePrice", axis=1)

#######################################
# MODEL: RANDOM FORESTS
#######################################

X = train_df.drop(['SalePrice', "Id"], axis=1)
y = np.log1p(train_df['SalePrice'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)

rf_model = RandomForestRegressor(random_state=42).fit(X_train, y_train)
y_pred = rf_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

# 0.05


y_pred = rf_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# 0.12

#######################################
# MODEL TUNING (RANDOM FORESTS)
#######################################

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 15],
             "n_estimators": [200, 500],
             "min_samples_split": [2, 5, 8]}

rf_model = RandomForestRegressor(random_state=42)
rf_cv_model = GridSearchCV(rf_model, rf_params, cv=10, n_jobs=-1, verbose=1).fit(X_train, y_train)
rf_cv_model.best_params_


#######################################
# FINAL MODEL
#######################################

rf_tuned = RandomForestRegressor(**rf_cv_model.best_params_).fit(X_train, y_train)

y_pred = rf_tuned.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

# 0.05

y_pred = rf_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# 0.12


#######################################
# FEATURE IMPORTANCE
#######################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_tuned, X_train, 40)


#######################################
# SONUCLARIN YUKLENMESI
#######################################

submission_df = pd.DataFrame()
submission_df['Id'] = test_df["Id"]

y_pred_sub = rf_tuned.predict(test_df.drop("Id", axis=1))
y_pred_sub = np.expm1(y_pred_sub)

submission_df['SalePrice'] = y_pred_sub

submission_df.head()
submission_df.tail()

submission_df.to_csv('submission_rf.csv', index=False)

#######################################
# DEGERLENDIRME
#######################################

# Kaggle Submission Score : 0.14
