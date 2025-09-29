# Retail Forecasting – Module 4

## Mô tả

Dự án này thực hiện phân tích toàn diện dữ liệu bán lẻ trực tuyến từ bộ dữ liệu "Online Retail II" của UCI Machine Learning Repository. Bộ dữ liệu chứa thông tin giao dịch từ một công ty bán lẻ quà tặng trực tuyến tại UK với hơn 1 triệu bản ghi từ tháng 12/2009 đến 12/2011.

### Các bước đã hoàn thành ✅

#### 🔄 **Bước 1: Data Cleaning & Preprocessing**
- Đọc và xử lý dữ liệu thô từ file `data_online_retail_II.csv`
- Xử lý missing values (Description, Customer ID)
- EDA cơ bản: thống kê mô tả, phân phối, missing values
- Tạo dữ liệu đã xử lý: `processed_data_gross.csv`

#### 📊 **Bước 2: Exploratory Data Analysis (EDA) Chi Tiết**
- Phân tích toàn diện với 15 bước chi tiết trên dữ liệu gross
- Schema analysis và data quality assessment
- Business cleaning và time series chuẩn hóa
- Phân tích mục tiêu Quantity và phân lớp nhu cầu theo SKU (ADI & CV²)
- Mùa vụ, giá cả, khách hàng, outliers analysis
- Time-based splits và leakage prevention
- Essential charts và feature suggestions
- Quality control dashboards

#### 🏗️ **Bước 3: Feature Engineering (leakage-safe, weekly W-MON)**
- Resample weekly theo `W-MON`, reindex liên tục từng SKU, fill-0 tuần không bán
- Snapshot TRAIN-only: `median_price_sku`, `p99_qty_sku`
- Đặc trưng thời gian dùng dữ liệu t−1 (tránh leakage):
  - Lags: `qty_lag_1..5,8,12`
  - Rolling/EMA: tính từ `qty_prev = Quantity.shift(1)`
  - Intermittency: `weeks_since_last_sale` từ `Quantity.shift(1)`, `zero_rate_*w` trên `qty_prev`
  - Giá/khuyến mãi: `price_prev = Price.shift(1)`, `price_index = price_prev/median_price_sku`,
    `d_price_pct_1w = price_prev vs price_prev_prev`, rolling/z-score trên `price_prev`
  - Cờ chất lượng: `extreme_qty_flag` dùng `qty_lag_1 > p99_qty_sku`, `price_jump_flag` dựa trên `price_prev`
- Tạo file: `fe_weekly_gross.csv`

#### 🤖 **Bước 4: Modeling (Two-stage với LightGBM & XGBoost)**
- Two-stage approach: Classification (has_sale) + Regression (quantity khi có sale)
- So sánh LightGBM vs XGBoost (baseline vs regularized)
- Split: train ≤ 2011-06-01, val 2011-06→2011-09, test 2011-09→2011-12
- Metrics:
  - Classification: Accuracy, Precision, Recall, F1 (VAL/TEST)
  - Regression: MAE, RMSE, WAPE, SMAPE (TEST)
- Visualizations:
  - Time series overlay (True vs dự báo) theo SKU trên VAL/TEST
  - Barplot tổng hợp metrics và phân phối residual trên TEST

#### 🔬 **Bước 5: XAI - Explainable AI với SHAP**
- SHAP analysis cho cả Classification và Regression models
- Global feature importance và feature impact distributions
- Dependency plots để hiểu mối quan hệ giữa features
- Local explanations cho các samples cụ thể
- Business insights từ feature importance analysis

## Cài đặt

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Hoặc sử dụng uv (nếu có)
uv sync

# Kiểm tra setup và dữ liệu
python test_notebook_setup.py

# Kích hoạt kernel venv (đã đăng ký)
jupyter kernelspec list
# Chọn kernel: Python (.venv Retail Forecasting)

# Chạy pipeline
jupyter notebook notebooks/03-Feature-Engineering.ipynb
jupyter notebook notebooks/04-Modeling.ipynb
```

## Cấu trúc dự án

```
project-4.2-Retail-Forecasting/
├── notebooks/
│   ├── 01-Data-Cleaning-and-EDA.ipynb        # Data cleaning & EDA cơ bản
│   ├── 02-EDA-Processed-Data-Gross.ipynb     # EDA chi tiết (15 bước)
│   ├── 03-Feature-Engineering.ipynb          # FE leakage-safe (weekly)
│   ├── 04-Modeling.ipynb                     # Two-stage LGBM/XGB + so sánh
│   ├── 05-XAI-Explainable-AI.ipynb          # SHAP analysis & business insights
│   └── reports/                              # Báo cáo và hình ảnh
├── data/
│   ├── raw/
│   │   └── data_online_retail_II.csv         # Dữ liệu gốc (1M+ records)
│   └── processed/
│       ├── processed_data_gross.csv          # Dữ liệu đã xử lý (1M+ records)
│       ├── fe_weekly_gross.csv               # Weekly FE (4863 SKUs, 106 weeks)
│       ├── snapshots/
│       │   └── snap_gross_train_stats.csv    # median_price & p99 by SKU (TRAIN-only)
│       └── predictions/
│           ├── test_predictions_lgb_xgb.csv  # Dự báo test (LGBM/XGB)
│           ├── val_predictions_lgb_xgb.csv   # Dự báo validation (LGBM/XGB)
│           ├── metrics_summary.csv
│           ├── classification_metrics_summary.csv
│           ├── classification_metrics_comparison_regularized.csv
│           └── regression_metrics_comparison_regularized.csv
├── requirements.txt                          # Dependencies
├── src/
│   └── utils/
│       └── plots.py                          # Hàm vẽ: SKU overlay & grid
└── README.md
```

## Kiểm tra và chạy

### 🔍 **Kiểm tra nhanh dữ liệu**
```bash
# Chạy kiểm tra toàn diện dữ liệu
python quick_checks.py

# Kiểm tra setup notebook
python test_notebook_setup.py
```

### 📊 **Chạy pipeline dự báo**

```bash
# 1) Data Cleaning & EDA cơ bản (tạo processed_data_gross.csv)
jupyter notebook notebooks/01-Data-Cleaning-and-EDA.ipynb

# 2) EDA chi tiết (15 bước phân tích toàn diện)
jupyter notebook notebooks/02-EDA-Processed-Data-Gross.ipynb

# 3) Feature Engineering (tạo fe_weekly_gross.csv)
jupyter notebook notebooks/03-Feature-Engineering.ipynb

# 4) Modeling hai bước (LGBM & XGB)
jupyter notebook notebooks/04-Modeling.ipynb

# 5) XAI - Explainable AI với SHAP
jupyter notebook notebooks/05-XAI-Explainable-AI.ipynb
```

## 📊 **Script EDA Chi Tiết (eda_detailed_explanations.py)**

Script này cung cấp báo cáo EDA hoàn chỉnh với 15 bước phân tích chi tiết:

### 🎯 **15 Bước EDA với chú thích đầy đủ:**

1. **Chọn phạm vi & hạt dữ liệu**
   - Cutoff date, forecast horizon, dominant country detection
   - Giải thích tại sao chọn weekly granularity

2. **Kiểm kê schema & chất lượng dữ liệu**
   - Schema table với %missing
   - Duplicates analysis với giải thích business meaning
   - Anomalies detection với recommendations

3. **Làm sạch nghiệp vụ (pre-clean)**
   - Technical codes, returns, bad prices
   - Business rules cho cleaning

4. **Chuẩn hóa chuỗi thời gian**
   - Weekly aggregation, zero-filling
   - Attributes calculation với ý nghĩa

5. **Khám phá mục tiêu (Quantity)**
   - Distribution analysis, quantiles
   - Rolling statistics với insights

6. **Phân lớp nhu cầu theo SKU**
   - ADI & CV² classification
   - Smooth/Intermittent/Erratic/Lumpy với recommendations

7. **Mùa vụ & chu kỳ**
   - Week-of-year heatmap analysis
   - Seasonal patterns insights

8. **Giá & khuyến mãi**
   - Price elasticity analysis
   - Promotion detection với business insights

9. **Khách hàng & giỏ hàng**
   - Customer loyalty analysis
   - Market basket patterns

10. **Outliers & anomalies**
    - Statistical outlier detection
    - Business rules cho outlier handling

11. **Phân khúc theo Country**
    - Geographic comparison
    - Bias detection và recommendations

12. **Split theo thời gian & chống rò rỉ**
    - Time-based splits
    - Leakage prevention checklist

13. **Essential Summary Charts**
    - Calendar heatmap Qty theo ngày/tuần (toàn bộ)
    - Top-50 SKU × tuần heatmap (chuẩn hóa theo SKU)
    - Scatter log Qty vs log Price (mẫu ngẫu nhiên có trọng số)
    - Bar zero-rate, returns-rate, promo-rate theo SKU (top-30)

14. **Feature Suggestions**
    - Feature importance preview chart (nếu chạy nhanh mô hình baseline) hoặc correlogram giữa candidate features (lag, rolling, promo, price change…)
    - Lag profile plot: tương quan Qty_t với Qty_{t−k} (k=1..12) theo nhóm SKU

15. **Deliverables & Quality Control**
    - Data quality dashboard: tiles hiển thị %NaN, %duplicates, #SKU, #weeks covered…
    - Anomaly audit table (SKU, tuần, loại flag, quyết định xử lý)
    - SKU summary table: lịch sử (tuần), ADI, CV², zero-rate, p95/p99 Qty, median price, promo-rate, returns-rate
    - Action tracker table: mỗi quyết định (keep/flag/winsorize/drop), lý do, ảnh hưởng %Qty/%revenue

### 🎉 **Kết quả thực tế từ script:**

- **Schema analysis** với missing values và data types
- **Duplicates detection** (1.035% logic duplicates)
- **Anomalies** (1.72% returns rate)
- **SKU classification** (61.5% lumpy, 37.9% erratic, 0.6% smooth)
- **Seasonal patterns** từ heatmap analysis
- **Price elasticity** và promotion effects
- **Geographic differences** giữa countries
- **Time splits** với leakage check
- **Essential summary charts** với heatmap và scatter plots
- **Feature suggestions** cho modeling pipeline
- **Quality control dashboards** và audit tables

## Sử dụng dữ liệu đã xử lý

### 1. **Dữ liệu chính đã xử lý**
File: `data/processed/processed_data_gross.csv`
- Chứa 1,002,894 bản ghi đã được làm sạch (sales only, gross data)
- Bao gồm các cột: Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country, InvoiceMonth, InvoiceQuarter, InvoiceYear, IsExistID, Country_encoded, Revenue

### 2. **Dữ liệu Feature Engineered (Weekly)**
File: `data/processed/fe_weekly_gross.csv`
- Dữ liệu weekly theo SKU (4,863 SKUs, 106 weeks)
- Bao gồm features leakage-safe: lags, rolling, EMA, price features, intermittency
- Target: `y` (quantity), `label_has_sale` (binary classification)

### 3. **Snapshot thống kê (TRAIN-only)**
File: `data/processed/snapshots/snap_gross_train_stats.csv`
- median_price_sku và p99_qty_sku theo từng SKU (tính từ TRAIN data only)
- Dùng để tránh leakage trong feature engineering

### 4. **Kết quả dự báo**
Files trong `data/processed/predictions/`:
- `test_predictions_lgb_xgb.csv`: Dự báo trên test set
- `val_predictions_lgb_xgb.csv`: Dự báo trên validation set
- Các file metrics summary và comparison

### 5. **Hướng dẫn sử dụng cho modeling**

```python
import pandas as pd
import numpy as np

# Đọc dữ liệu FE
fe = pd.read_csv('data/processed/fe_weekly_gross.csv', parse_dates=['InvoiceDate'])
fe['label_has_sale'] = (fe['y'] > 0).astype(int)

# No-leakage features (loại bỏ các cột có thể gây leakage)
drop_cols = {'InvoiceDate','StockCode','Country','y','label_has_sale','Quantity','Price'}
feature_cols = [c for c in fe.columns if c not in drop_cols]

# Time-based splits (leakage-safe)
TRAIN_END = pd.Timestamp('2011-06-01')
VAL_END = pd.Timestamp('2011-09-01')

train_idx = fe['InvoiceDate'] < TRAIN_END
val_idx = (fe['InvoiceDate'] >= TRAIN_END) & (fe['InvoiceDate'] < VAL_END)
test_idx = fe['InvoiceDate'] >= VAL_END

# Training data
X_train = fe.loc[train_idx, feature_cols]
y_train_cls = fe.loc[train_idx, 'label_has_sale']
y_train_reg = fe.loc[train_idx, 'y']

# Validation/Test data
X_val = fe.loc[val_idx, feature_cols]
y_val_cls = fe.loc[val_idx, 'label_has_sale']
X_test = fe.loc[test_idx, feature_cols]
y_test_cls = fe.loc[test_idx, 'label_has_sale']
```

## Yêu cầu hệ thống

- Python >= 3.8
- RAM: Tối thiểu 8GB (khuyến nghị 16GB cho dữ liệu lớn)
- Storage: Tối thiểu 5GB để lưu trữ dữ liệu và kết quả

## Các thư viện chính

- **pandas**: Xử lý và phân tích dữ liệu
- **numpy**: Tính toán số học
- **matplotlib/seaborn/plotly**: Trực quan hóa
- **scikit-learn**: Machine learning và evaluation metrics
- **lightgbm**: Gradient boosting framework
- **xgboost**: Gradient boosting framework (native DMatrix API)
- **jupyter**: Môi trường phân tích tương tác
