# %% [markdown]
# # Manufacturing Data Science 製造數據科學 - Assignment 2
# > R10725012 呂晟維

# %% [markdown]
# ### Q1 (20%)

# %% [markdown]
# #### Q1-(a) 
# **試簡述何謂維度的詛咒？試列舉一案例說明**
# 
# Ans:  
# 模型的參數數量與模型本身的複雜度與資料集的特徵維度呈正相關，
# 模型的參數數量越多，使模型達成收斂需要的資料筆數也就越多，當樣本數不足時會導致
# 1. 花很長時間學習但演算法不易收斂
# 2. 收斂時出現多重解 (multiple solutions) 或過度配適
# 
# 簡而言之，特徵從零變多模型的預測能力一開始會提升，當過了最適的特徵個數，預測績效立即呈現指數遞減。

# %% [markdown]
# #### Q1-(b)
# **避免維度詛咒的方法有哪些？**  
# Ans  
# 1. 避免使用過多特徵，僅使用最適的特徵個數數量的作為訓練資料。
# 2. 檢查特徵間是否有共線性關係，若有則整合或剔除具共線性關係的特徵們。

# %% [markdown]
# #### Q1-(c)
# 試找一個開放數據 (e.g. Kaggle開放數據 或第一次作業紅酒數據集 )並選一種方法 (e.g. 線性迴歸或決策樹 )，用模擬方法 固定樣本數但逐步增加變數個數， 試著重新繪製圖 3.12， 呈現維度與預測 (或分類 )績效間的關係。

# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
# import statsmodels.stats.api as sms
# import scipy.stats as stats

# %%
# import wine data, x = f0~f27, y = grade
furnace_data = pd.read_csv("../MDS_Assignment1/MDS_Assignment1_furnace.csv")
print(furnace_data.shape)
display(furnace_data.head())

# %%
furnace_X = furnace_data.iloc[:, :-1] # params
furnace_y = furnace_data.iloc[:, -1] # label
furnace_X_const = sm.add_constant(furnace_X) # 做線性回歸前 要手動加上截距(常數項)
furnace_X_const = furnace_X_const.rename(columns={"const": "Intercept"})
display(furnace_X_const.head())
print('labels are:', set(furnace_y))

# %%
# (1) 整體資料先做線性迴歸；
# (2) 依 p value 小至大排序選出重要變數；
furnace_model = sm.OLS(furnace_y, furnace_X_const)
furnace_result = furnace_model.fit()
# print(furnace_result.summary())
print('rsquared:', furnace_result.rsquared, '\nrsquared_adj:', furnace_result.rsquared_adj)

furnace_pvalues = furnace_result.pvalues
print(furnace_pvalues.sort_values().head(3)) # p 越小影響力越大
print(furnace_pvalues.sort_values().tail(3)) # p 越大影響力越小

sorted_pvalues = list(furnace_pvalues.sort_values().keys())
print(f"sorted_pvalues: {sorted_pvalues}")

# %%
# 這個 function 吃全域變數喔

def make_formula(numOfParam: int) -> str:
    ''' 1 <= numOfParam <= 28 \n
    no need to add constant \n
    example output formula with numOfParam = 4:
    `grade ~ f0 + f1 + f2 + f3` '''
    if numOfParam < 1:
        numOfParam = 1
    if numOfParam > 28:
        numOfParam = 28
    param_lst = sorted_pvalues[1:]

    s = f"{furnace_data.columns[-1]} ~" # label
    first_param = True
    for x in param_lst[:numOfParam]:
        if first_param:
            s += f' {x}' # s += ' param'
            first_param = False
        else:
            s += f' + {x}' # s += ' + param'
    return s

make_formula(1) # min param
make_formula(28) # max param

# %%
# (3) 將重要的變數一個個依序放入迴歸並計算 adjusted R2 作為預測準確度
# formula string format: 'Label ~ param1 + param2 ...'
numOfTest = len(sorted_pvalues[1:])
rsquared_adj_lst = []

for i in range(1, numOfTest+1):
    formula_str = make_formula(i)
    result = smf.ols(formula=formula_str, data=furnace_data).fit()
    print(f'rsquared_adj with {i} params: {result.rsquared_adj}')
    rsquared_adj_lst.append(result.rsquared_adj)

plt.plot(rsquared_adj_lst)
plt.xlabel('Dimension') 
plt.ylabel('Rsquared adj')
plt.xticks(np.arange(numOfTest), np.arange(1, numOfTest+1))
plt.grid()

# result = smf.ols(formula=formula_str, data=furnace_data).fit()
# print('rsquared:', result.rsquared, '\nrsquared_adj:', result.rsquared_adj)

# %% [markdown]
# ### Q2 (20%)

# %% [markdown]
# #### Q2-(a)
# 試找一個開放數據 (e.g. Kaggle開放數據 )，您會用什麼方法來確認資料品質的好壞?試操作一次並說明其細節

# %%
# https://www.kaggle.com/datasets/whenamancodes/covid-19-coronavirus-pandemic-dataset
# https://www.kaggle.com/datasets/segunadedipe/nigerian-car-prices
df = pd.read_csv("Nigerian_Car_Prices.csv")
df = df.iloc[:, 1:]
print(df.shape)
df.head()

# %%
# Missing value rate
# ref https://datatofish.com/check-nan-pandas-dataframe/
numOfNanCell = df.isnull().sum().sum()
numOfTotalCall = df.shape[0]*df.shape[1]
missRate = numOfNanCell / numOfTotalCall
print(f"Missing value rate is {100*missRate:.2f}%. ({numOfNanCell} out of {numOfTotalCall})")

# df.isnull().sum().plot(kind='bar')
ax = df.isnull().sum().plot.bar(rot=45)
for container in ax.containers:
    ax.bar_label(container)

# %%
df['Make'].value_counts(normalize=True, sort=True).head()

# %%
# 進行獨行性測試(OLS建模)之前，需要補空值，把類別轉成數值
def format_float(x: str):
    return x.replace(',','')  

df['Price'] = df['Price'].apply(lambda x: format_float(x))
       
# 數值資料填 mean
for col_name in ['Year of manufacture', 'Mileage', 'Engine Size', 'Price']:
    df[col_name] = pd.to_numeric(df[col_name])
    df[col_name].fillna(value=df[col_name].mean(), inplace=True)
# df.fillna(value=0, inplace=True)

# 類別資料轉成 0,1,2 空值 -1
for col_name in ['Make', 'Condition', 'Fuel', 'Transmission', 'Build']:
    # df[col_name] = df[col_name].astype('category')
    df[col_name] = pd.factorize( df[col_name] )[0]
df.head()

# %%
# Independence by durbin_watson
from statsmodels.formula.api import ols
from statsmodels.stats.stattools import durbin_watson
import pandas as pd
import numpy as np

# df = pd.DataFrame(np.random.standard_normal((500,4)))
# df.columns = ["rating", "points", "assists", "rebounds"]

# fit multiple linear regression model
model = ols("Make ~ Q('Year of manufacture') + Condition + Mileage + Q('Engine Size') + \
    Fuel + Transmission + Price + Build", data=df)
res = model.fit()

dw = durbin_watson(res.resid)
print(f"Durbin-Watson: {dw}")

# %%
# Entropy of 'Make', we can see the entropy is high, the infomation in rich enough.
# ref https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python
from math import log, e
def entropy3(labels, base=None):
  vc = pd.Series(labels).value_counts(normalize=True, sort=False)
  print(f'number of brand: {len(vc)}')
  base = e if base is None else base
  return -(vc * np.log(vc)/np.log(base)).sum()

entropy3(df['Make'])

# %% [markdown]
# #### Q2-(b)
# 公司或您 是否有現存方法來進行資料品質的確認？如果有(或沒有)，試依您的角度說明(或建議)確認資料品質的標準作業流程 (i.e. SOP)。

# %% [markdown]
# #### Q2-(c)
# 試建議三個可能衡量數據品質的量化指標 (i.e. KPIs)。

# %% [markdown]
# ### Q3 (20%) 在數據科學分析架構中的決策支援階段

# %% [markdown]
# #### Q3-(a)
# 什麼是模型的適應性與擴充性?

# %%


# %% [markdown]
# #### Q3-(b)
# 在AI專案中(可根據第一題的開放數據與模型)，就您所使用的數據與建構預測模型是否具備適應性與擴充性?為什麼?該如何改善與調整?

# %% [markdown]
# #### Q4 (10%) 
# 遺漏值填補的方法包括了統計量填補 、預測式與生成式填補

# %% [markdown]
# #### Q4-(a)
# 試說明這些方法分別適用於什麼樣情形

# %%


# %% [markdown]
# #### Q4-(b)
# 為什麼某特徵存在大量遺漏值不宜直接刪除？

# %% [markdown]
# ### Q5 (30%) 
# 在 UCI Machine Learning Repository 開放數據中包含了一個鋼板缺陷數據 (steel plates faults dataset https://archive.ics.uci.edu/ml/datasets/steel+plates+faults)，一共包含了1,941個觀測值，而每個觀測值具有 27個特徵以及作為目標值的7種缺陷。試挑選出凹凸不平(Bumps)以及刮痕(K_Scratch)兩種缺陷進行分析

# %% [markdown]
# #### Q5-(1)
# 試將羅吉斯迴歸分析的結果呈現如下表，並試著解釋任一特徵與目標值之間的關係。

# %%
# ! pip install openpyxl

# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
xlxs_path = "./MDS_Assignment2_Steelplates.xlsx"
df = pd.read_excel(xlxs_path, sheet_name="Var_Name", header=None)
columns = df.iloc[:,0].to_numpy()
columns

df = pd.read_excel(xlxs_path, sheet_name="Faults", header=None, names=columns)
df

# %%
print('特徵27種:',columns[:27])
print('標籤 7種:',columns[27:])

# %%
# 只取出指定的兩種 label 做分析
# # record_ids = (df["K_Scatch"] == 1) | (df["Bumps"] == 1) 
# df_filter = df.loc[record_ids]

x = df.iloc[:, 0:27]
# x_bump = df.iloc[:, 0:27]
y_bump = df["Bumps"]
print('num of bump samples:', len(df[df["Bumps"] == 1]), 'out of', len(df))

# x_scatch = df.iloc[:, 0:27]
y_scatch = df["K_Scatch"]
print('num of scatch samples:', len(df[df["K_Scatch"] == 1]), 'out of', len(df))

# %% [markdown]
# ##### Bumps Logistic Model

# %%
# Bumps = 1, other = 0 in this condition
s_x = sm.add_constant(x)
model_bump = sm.MNLogit(y_bump, s_x).fit()
print(model_bump.summary())

# %%
# confusion matrix of class Bumps prediction
# model_bump.pred_table() 一行搞定版
pred = np.array(model_bump.predict(s_x) > 0.5, dtype=float)
table = np.histogram2d(y_bump, pred[:,1], bins=2)[0]
table

# %% [markdown]
# ##### K_Scatch Logistic Model

# %%
# K_Scatch = 1, other = 0 in this condition
s_x = sm.add_constant(x)
model_scatch = sm.MNLogit(y_scatch, s_x).fit()
print(model_scatch.summary())

# %%
# confusion matrix of class K_Scatch prediction
model_scatch.pred_table()

# %% [markdown]
# #### Q5-(2) 
# 基於上述(1)的 結果，將上述特徵以t-value進行排序後，哪些特徵的迴 歸係數在統計上是顯著的呢(p-value<0.01)?

# %%
p_values_bump = model_bump.pvalues
p_values_bump = p_values_bump[p_values_bump[0] < 0.01]
p_values_bump.sort_values(by=0)

# %%
p_values_scatch = model_scatch.pvalues
p_values_scatch = p_values_scatch[p_values_scatch[0] < 0.01]
p_values_scatch.sort_values(by=0)

# %% [markdown]
# #### Q5-(3) 
# 試問配適一個羅吉斯迴歸模型是否合適？試若配適不佳，試說明其可能的原因為何？

# %% [markdown]
# #### Q5-(4)
# 試問配適一個線性判別分析模型是否合適？若配適不佳，試說明其可能的原因為何？

# %%
# 生成模型可一次預測多類別，將資料做成多類別的
# 0: 正常無缺陷, 1: Bumps, 2: K_Scatch
K_Scatch_ids = df.index[df['K_Scatch'] == 1].tolist()
Bumps_ids = df.index[df['Bumps'] == 1].tolist()
y_true = [0]*len(df)

for i in Bumps_ids:
    y_true[i] = 1
for i in K_Scatch_ids:
    y_true[i] = 2

from collections import Counter
Counter(y_true)

# %%
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis()
clf.fit(x, y_true)

# %%
from sklearn.metrics import classification_report
y_pred = clf.predict(x)
report = classification_report(y_true, y_pred, target_names=['Noraml','Bumps','K_Scatch'])
print(report)

# %% [markdown]
# #### Q5-(5)
# 試問配適一個二次判別分析模型是否合適？若配適不佳，試說明其可能的原因為何？  
# https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis

# %%
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
clf = QuadraticDiscriminantAnalysis()
clf.fit(x, y_true)

from sklearn.metrics import classification_report
y_pred = clf.predict(x)
report = classification_report(y_true, y_pred, target_names=['Noraml','Bumps','K_Scatch'])
print(report)

# %%



