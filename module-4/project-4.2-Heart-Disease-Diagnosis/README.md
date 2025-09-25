# Online Retail Data Analysis Project

Phân tích dữ liệu bán lẻ trực tuyến - Dự án Module 4

## Mô tả

Dự án này thực hiện phân tích toàn diện dữ liệu bán lẻ trực tuyến từ bộ dữ liệu "Online Retail II" của UCI Machine Learning Repository. Bộ dữ liệu chứa thông tin giao dịch từ một công ty bán lẻ quà tặng trực tuyến tại UK với hơn 1 triệu bản ghi từ tháng 12/2009 đến 12/2011.

### Các bước đã hoàn thành ✅

#### 🔄 **Bước 1: Data Cleaning & Preprocessing**
- Xử lý dữ liệu thô từ file `data_online_retail_II.csv`
- Loại bỏ các bản ghi null và không có ý nghĩa
- Kiểm tra và xử lý các giá trị ngoại lệ
- Tạo các đặc trưng mới từ dữ liệu gốc

#### 📊 **Bước 2: Exploratory Data Analysis (EDA)**
- Phân tích thống kê mô tả
- Trực quan hóa phân phối dữ liệu
- Phân tích tương quan giữa các biến
- Tạo các báo cáo phân tích chi tiết

#### 🏗️ **Bước 3: Feature Engineering**
- Mã hóa các biến phân loại (Country, Invoice status)
- Tạo biến `Revenue` = `Quantity` × `Price`
- Trích xuất thông tin thời gian (Month, Quarter, Year)
- Tạo bảng phân tích nhóm theo `StockCode` với:
  - Thống kê giá (mean, min, max, std)
  - Thống kê nhu cầu theo tháng, quý, năm

#### 📈 **Bước 4: Data Visualization & Reporting**
- Tạo 7 báo cáo phân tích trực quan:
  - Phân tích hủy đơn hàng
  - Phân tích khách hàng
  - Phân tích theo thời gian
  - Phân tích giá cả
  - Phân tích doanh thu
  - Heatmap dữ liệu thiếu
  - Phân tích chuỗi thời gian

### 🔄 **Bước tiếp theo cần thực hiện** ⏭️

#### 🤖 **Bước 5: Machine Learning Modeling** (Chưa thực hiện)
- Xây dựng mô hình dự đoán:
  - Dự đoán doanh thu theo sản phẩm
  - Phân loại khách hàng
  - Dự đoán xu hướng mua hàng
  - Phân cụm khách hàng và sản phẩm
- Sử dụng các thuật toán:
  - Regression models
  - Classification models
  - Clustering algorithms
  - Time series forecasting

#### 🚀 **Bước 6: Deployment & Application** (Chưa thực hiện)
- Xây dựng dashboard tương tác
- Tạo API cho dự đoán
- Deploy mô hình lên production
- Tích hợp với hệ thống hiện tại

## Cài đặt

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Hoặc sử dụng uv (nếu có)
uv sync

# Chạy Jupyter Notebook để xem phân tích
jupyter notebook notebooks/01-Data-Cleaning-and-EDA.ipynb
```

## Cấu trúc dự án

```
project-retail-analysis/
├── notebooks/
│   └── 01-Data-Cleaning-and-EDA.ipynb    # Notebook phân tích chính
│   └── reports/                          # Báo cáo và hình ảnh
├── data/
│   ├── raw/
│   │   └── data_online_retail_II.csv     # Dữ liệu gốc
│   └── processed/
│       ├── processed_data.csv            # Dữ liệu đã xử lý
│       └── data_stockcode_grouped.csv    # Dữ liệu nhóm theo StockCode
├── src/                                  # Source code (sẽ phát triển)
├── requirements.txt                      # Dependencies
└── README.md
```

## Sử dụng dữ liệu đã xử lý

### 1. **Dữ liệu chính đã xử lý**
File: `data/processed/processed_data.csv`
- Chứa 1,020,468 bản ghi đã được làm sạch
- Bao gồm các cột: Invoice, StockCode, Quantity, InvoiceDate, Price, Customer ID, Invoice_cancelled, InvoiceMonth, InvoiceQuarter, InvoiceYear, IsExistID, Country_encoded, Revenue

### 2. **Dữ liệu phân tích nhóm**
File: `data/processed/data_stockcode_grouped.csv`
- Chứa 4,869 mã sản phẩm (StockCode) với thống kê chi tiết
- Bao gồm: num_price_level, price_mean, price_min, price_max, price_std, price_mode, quantity_monthly, quantity_quarterly, quantity_yearly

### 3. **Hướng dẫn cho bước tiếp theo**

```python
import pandas as pd

# Đọc dữ liệu đã xử lý
df_processed = pd.read_csv('data/processed/processed_data.csv')
df_grouped = pd.read_csv('data/processed/data_stockcode_grouped.csv')

# Merge dữ liệu để có đầy đủ thông tin
df_ml = df_processed.merge(df_grouped, on='StockCode', how='left')

# Bây giờ bạn có thể:
# 1. Chia train/test split
# 2. Xây dựng features cho ML
# 3. Train các mô hình dự đoán
# 4. Đánh giá performance
```

## Yêu cầu hệ thống

- Python >= 3.8
- RAM: Tối thiểu 8GB (khuyến nghị 16GB cho dữ liệu lớn)
- Storage: Tối thiểu 5GB để lưu trữ dữ liệu và kết quả

## Các thư viện chính

- **pandas**: Xử lý và phân tích dữ liệu
- **numpy**: Tính toán số học
- **matplotlib/seaborn/plotly**: Trực quan hóa
- **scikit-learn**: Machine learning (sẽ sử dụng ở bước tiếp theo)
- **jupyter**: Môi trường phân tích tương tác
