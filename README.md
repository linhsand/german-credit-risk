# German Credit Risk Prediction

## 📌 Giới thiệu
Project dự đoán rủi ro tín dụng dựa trên German Credit Dataset.  
Bao gồm các phần:
- EDA.ipynb: phân tích dữ liệu
- preprocess.py: tiền xử lý & feature engineering
- train.py: huấn luyện mô hình RandomForest với pipeline
- models/model.joblib: mô hình đã huấn luyện

## 🚀 Cách chạy

### 1. Tạo môi trường
```bash
python -m venv .venv
source .venv/bin/activate   # hoặc .venv\Scripts\activate với Windows
pip install -r requirements.txt
