# 📊 Pandas Data Analysis Tutorial
## 資料科學基礎 - Pandas 完整學習筆記

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-green.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific_Computing-orange.svg)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-red.svg)](https://jupyter.org/)

### 📋 專案概述

本專案是一個全面的Pandas資料分析學習筆記，涵蓋了從基礎到進階的Pandas操作技巧。透過實際的程式碼範例和練習，系統性地學習Python資料科學生態系統中最重要的資料處理工具。

### 🎯 學習目標

- ✅ **Pandas基礎操作**：Series和DataFrame的創建與操作
- ✅ **資料讀取與寫入**：多種檔案格式的I/O操作
- ✅ **資料清理與處理**：缺失值處理、資料型態轉換
- ✅ **資料篩選與查詢**：進階索引與條件篩選技術
- ✅ **資料聚合與分組**：GroupBy操作與統計分析
- ✅ **資料合併與連接**：Join、Merge等資料整合技術

### 📚 學習內容架構

#### **🔧 核心資料結構**
```python
# 1. Series 基礎操作
s = pd.Series([1, 3, 5, np.nan, 6, 8])

# 2. DataFrame 創建與操作  
dates = pd.date_range("20130101", periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))
```

#### **📊 資料操作技能**

**1️⃣ 資料創建與索引**
- 時間序列索引生成
- 多層次索引設計
- 自定義欄位命名
- 資料型態設定

**2️⃣ 資料篩選與選擇**
- 條件篩選 (`df[df > 0]`)
- 布林索引操作
- 位置索引與標籤索引
- 多條件邏輯組合

**3️⃣ 資料處理與清理**
- 缺失值檢測與處理
- 資料型態轉換
- 重複值移除
- 異常值識別

**4️⃣ 統計分析與聚合**
- 描述性統計計算
- GroupBy分組操作
- 聚合函數應用
- 透視表分析

**5️⃣ 資料合併與重塑**
- DataFrame合併技術
- 資料連接操作
- 長寬格式轉換
- 階層資料處理

**6️⃣ 檔案I/O操作**
- CSV檔案讀寫
- Excel檔案處理
- 多工作表操作
- 資料格式轉換

### 📁 專案結構

```
Pandas_Learning_Project/
│
├── README.md                        # 專案說明文件
├── 142216015_劉玳如_hw01.ipynb     # 主要學習筆記
├── 142216015_劉玳如_hw02.pptx      # 學習成果簡報
│
├── Data_Examples/                   # 練習資料集
│   ├── sample_data.csv
│   ├── foo.xlsx
│   └── test_data.json
│
├── Exercise_Solutions/              # 練習解答
│   ├── basic_operations.py
│   ├── data_manipulation.py
│   └── advanced_analysis.py
│
└── Resources/                       # 學習資源
    ├── pandas_cheatsheet.pdf
    └── reference_guide.md
```

### 🛠️ 環境設置

#### **必要套件**
```python
# 核心資料科學套件
import numpy as np           # 數值計算基礎
import pandas as pd          # 資料分析主工具

# 支援套件  
import matplotlib.pyplot as plt  # 視覺化
import seaborn as sns           # 進階視覺化
```

#### **版本需求**
```bash
Python >= 3.8
pandas >= 1.3.0  
numpy >= 1.21.0
jupyter >= 1.0.0
```

### 🚀 快速開始

#### **1. 環境安裝**
```bash
# 使用conda安裝
conda install pandas numpy jupyter

# 或使用pip安裝  
pip install pandas numpy jupyter matplotlib seaborn
```

#### **2. 啟動學習環境**
```bash
# 啟動Jupyter Notebook
jupyter notebook

# 開啟主要學習檔案
# 142216015_劉玳如_hw01.ipynb
```

#### **3. 基礎操作驗證**
```python
import pandas as pd
import numpy as np

# 測試環境是否正常
print(pd.__version__)
print(np.__version__)
```

### 📖 核心學習模組

#### **Module 1: 基礎資料結構**
```python
# Series 操作基礎
series_basic = pd.Series([1, 3, 5, np.nan, 6, 8])
print(series_basic.describe())

# DataFrame 創建方法
df_basic = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': pd.date_range('20130101', periods=4),
    'C': pd.Series([1, 3, 5, 7], dtype='float32'),
    'D': np.array([3] * 4, dtype='int32')
})
```

#### **Module 2: 進階索引技術**
```python
# 日期索引創建
dates = pd.date_range("20130101", periods=6)
df = pd.DataFrame(np.random.randn(6, 4), 
                  index=dates, columns=list("ABCD"))

# 條件篩選與布林索引
filtered_data = df[df > 0]
complex_filter = df[(df['A'] > 0) & (df['B'] < 0)]
```

#### **Module 3: 資料合併與連接**
```python
# Left Join 操作
left = pd.DataFrame({"key": ["foo", "foo"], "lval": [1, 2]})
right = pd.DataFrame({"key": ["foo", "foo"], "rval": [4, 5]})
merged = pd.merge(left, right, on="key")

# Concatenate 操作
pieces = [df[:3], df[3:7], df[7:]]
concatenated = pd.concat(pieces)
```

#### **Module 4: 檔案I/O實務**
```python
# Excel 檔案處理
df.to_excel("output.xlsx", sheet_name="Sheet1")
read_data = pd.read_excel("foo.xlsx", "Sheet1", 
                         index_col=None, na_values=["NA"])

# CSV 檔案操作
df.to_csv("data.csv", index=False)
csv_data = pd.read_csv("data.csv")
```

### 💡 重要概念與技巧

#### **🔍 資料檢視技巧**
```python
# 快速資料概覽
df.head()           # 前5行
df.tail(3)          # 後3行
df.index            # 索引資訊
df.columns          # 欄位名稱
df.describe()       # 統計摘要
df.info()          # 資料框架資訊
```

#### **🛠️ 常用資料處理**
```python
# 缺失值處理
df.isnull().sum()   # 統計缺失值
df.dropna()         # 移除缺失值
df.fillna(value=5)  # 填補缺失值

# 資料篩選
df.loc['2013-01-01':'2013-01-03']  # 日期範圍
df.iloc[3:5, 0:2]                  # 位置索引
df[df.A > df.C]                    # 條件篩選
```

#### **📊 統計與聚合**
```python
# 基本統計
df.mean()          # 平均值
df.std()           # 標準差
df.apply(np.cumsum) # 累積和

# 分組操作
df.groupby('A').sum()        # 分組加總
df.groupby(['A', 'B']).mean() # 多欄位分組
```

### ⚠️ 常見問題與解決方案

#### **🐛 典型錯誤處理**

**1. Series 布林值判斷錯誤**
```python
# ❌ 錯誤做法
if pd.Series([False, True, False]):
    print("I was true")
    
# ✅ 正確做法
if pd.Series([False, True, False]).any():
    print("I was true")
```

**2. 檔案編碼問題**
```python
# ✅ 指定編碼讀取
df = pd.read_csv('data.csv', encoding='utf-8')
```

**3. 記憶體優化**
```python
# ✅ 指定資料型態節省記憶體
df['category_col'] = df['category_col'].astype('category')
```

### 🎯 實務應用場景

#### **📈 商業分析應用**
```python
# 銷售資料分析
sales_data = pd.read_excel('sales.xlsx')
monthly_sales = sales_data.groupby('month')['amount'].sum()
top_products = sales_data.nlargest(10, 'revenue')
```

#### **🔬 科學研究應用**
```python
# 實驗資料處理
experiment_data = pd.read_csv('experiment.csv')
grouped_results = experiment_data.groupby('treatment').agg({
    'measurement': ['mean', 'std', 'count']
})
```

#### **💰 金融資料分析**
```python
# 股價資料分析
stock_prices = pd.read_csv('stocks.csv', parse_dates=['date'])
stock_prices.set_index('date', inplace=True)
returns = stock_prices.pct_change()
rolling_mean = stock_prices.rolling(window=30).mean()
```

### 📚 進階學習路徑

#### **🎓 下一步學習建議**
1. **視覺化整合**：Matplotlib, Seaborn, Plotly
2. **機器學習銜接**：Scikit-learn資料前處理
3. **大數據處理**：Dask, Vaex等大數據工具
4. **時間序列分析**：進階時間序列操作
5. **效能優化**：Pandas最佳化技巧

#### **📖 推薦學習資源**
- Pandas官方文檔
- "Python for Data Analysis" - Wes McKinney
- Kaggle Learn課程
- DataCamp Pandas軌道

### 🔧 開發最佳實踐

#### **💻 程式碼風格**
```python
# ✅ 良好的程式碼風格
import pandas as pd
import numpy as np

def process_data(df):
    """
    處理資料的標準流程
    """
    # 移除重複值
    df_clean = df.drop_duplicates()
    
    # 處理缺失值
    df_clean = df_clean.fillna(df_clean.mean())
    
    return df_clean
```

#### **📊 效能優化技巧**
```python
# ✅ 向量化操作代替迴圈
df['new_col'] = df['col1'] * df['col2']  # 快

# ❌ 避免使用迴圈
for idx, row in df.iterrows():  # 慢
    df.loc[idx, 'new_col'] = row['col1'] * row['col2']
```

### 🚀 未來發展方向

#### **🔬 進階技能規劃**
- [ ] **Pandas 2.0新功能**：學習最新版本特性
- [ ] **分散式計算**：Dask整合應用
- [ ] **資料庫整合**：SQL與Pandas結合
- [ ] **即時資料處理**：串流資料分析
- [ ] **自動化流程**：資料處理管道建立

#### **💼 職涯應用方向**
- **資料科學家**：進階分析與建模
- **資料工程師**：ETL流程設計  
- **商業分析師**：商務洞察分析
- **研究分析師**：學術研究支援

### 📈 學習成果展示

這個專案展示了以下核心能力：

#### **🎯 技術能力**
- **Pandas熟練度**：從基礎到進階的完整掌握
- **資料處理技巧**：實務導向的操作經驗
- **程式設計能力**：清晰的程式邏輯與結構
- **問題解決能力**：錯誤除錯與最佳化

#### **📊 實務經驗**
- **多元資料格式**：CSV、Excel、JSON等格式處理
- **複雜資料操作**：合併、重塑、聚合等高階技巧
- **效能優化意識**：記憶體使用與運算效率考量
- **最佳實務應用**：工業級程式碼標準
