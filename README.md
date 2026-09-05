# MegaRAG Vietnamese Document Image Reasoning

## 1. Giới thiệu
Dự án ứng dụng **MegaRAG** kết hợp **Qwen3-VL-8B** và **MMKG (Multimodal Knowledge Graph)** để trả lời câu hỏi dựa trên ảnh tài liệu tiếng Việt.
Mã nguồn này là sản phẩm cuối cùng của Jupyter Notebook, đã được tinh chỉnh (patch) để chạy hoàn toàn local trên Kaggle/Colab mà không cần API OpenAI.

## 2. Cấu trúc thư mục nộp bài
```text
submission_pack/
├── MegaRAG/                 # Source code MegaRAG (đã patch dùng Qwen local & tiếng Việt)
│   ├── egs/                 # Chứa các script OCR, build MMKG, Query, Baseline, Judge
│   ├── megarag/             # Core library
│   └── demo_app/            # Giao diện demo Streamlit
│       ├── app.py
│       └── requirements.txt
├── outputs/                 # Kết quả đánh giá (predictions, metrics, judges, demo_results)
│   ├── metrics.json
│   ├── predictions.csv
│   └── results/
└── README.md                # File hướng dẫn này
```
## 3. Hướng dẫn chạy chương trình Demo (Streamlit)
Yêu cầu hệ thống  
Python 3.10+  
GPU NVIDIA với VRAM >= 16GB (để load Qwen3-VL-8B).  
Đã chạy Jupyter Notebook để xuất ra file MMKG và outputs/demo_results.jsonl.  
Các bước chạy  
Giải nén file FINAL_SUBMISSION.zip.  
Di chuyển vào thư mục demo:  
```bash
   cd MegaRAG/demo_app
   pip install -r requirements.txt
```
Chạy ứng dụng Streamlit:
```bash
   streamlit run app.py
```
Giao diện web tại http://localhost:8501. Nhập câu hỏi tiếng Việt hoặc bấm vào các câu hỏi mẫu để xem kết quả truy vấn từ MMKG.

## 4. Tài liệu tham khảo
Source code gốc MegaRAG: https://github.com/AI-Application-and-Integration-Lab/MegaRAG
Bài báo: https://aclanthology.org/2026.acl-long.2218.pdf
Dataset tiếng Việt: https://huggingface.co/datasets/trannhiem/TranNhiem-Vietnamese-DocumentImage-Reasoning
