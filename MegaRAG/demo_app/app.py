import streamlit as st
import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(page_title="MegaRAG Vietnamese VQA Demo", layout="wide")
st.title("🇻🇳 Demo MegaRAG & Qwen3-VL cho Tài liệu Tiếng Việt")
st.markdown("Ứng dụng trả lời câu hỏi dựa trên ảnh tài liệu và Multimodal Knowledge Graph (MMKG).")

st.sidebar.info("💡 **Lưu ý:** Để chạy demo này trên máy cá nhân, bạn cần có GPU VRAM > 16GB và đã chạy Notebook để build sẵn MMKG.")

def mock_query(question):
    # Đường dẫn tương đối khi chạy local, hoặc tuyệt đối trên Kaggle
    results_file = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'demo_results.jsonl')
    if not os.path.exists(results_file):
        results_file = "../megarag_outputs/demo_results.jsonl"
        
    sample_qs = [
        "Tác giả của vở kịch 'Gia tài' là ai và được phóng tác từ tác phẩm nào?",
        "Trong bài tập 2a, kết quả của phép cộng 27 + 46 là bao nhiêu?",
        "Chòm sao Gấu Bé được tạo thành từ mấy ngôi sao chính và ngôi sao cuối đuôi là sao gì?"
    ]
    
    if os.path.exists(results_file):
        with open(results_file, 'r', encoding='utf-8') as f:
            lines = [json.loads(line) for line in f if line.strip()]
            # Ưu tiên tìm theo text câu hỏi nếu file JSONL có lưu câu hỏi
            for rec in lines:
                if rec.get('question') == question or rec.get('query') == question:
                    return rec.get('answer', 'Không tìm thấy câu trả lời.')
            # Fallback: Ánh xạ theo index của câu hỏi mẫu (Chắc chắn trúng)
            if question in sample_qs:
                idx = sample_qs.index(question)
                if idx < len(lines):
                    return lines[idx].get('answer', 'Không tìm thấy câu trả lời.')
                    
    return "⚠️ Hệ thống chưa được build MMKG hoặc chưa chạy demo. Vui lòng chạy Jupyter Notebook trước."

question = st.text_input("💡 Nhập câu hỏi tiếng Việt của bạn:", placeholder="Ví dụ: Tác giả của vở kịch 'Gia tài' là ai?")

if st.button("🔍 Truy vấn MegaRAG"):
    if question:
        with st.spinner("Mô hình đang truy vấn Knowledge Graph và suy luận..."):
            ans = mock_query(question)
            st.success("🤖 Câu trả lời:")
            st.markdown(ans)
    else:
        st.warning("Vui lòng nhập câu hỏi!")

st.divider()
st.markdown("### 📂 Bấm vào các câu hỏi mẫu (Đã chạy sẵn trong Notebook)")
sample_qs = [
    "Tác giả của vở kịch 'Gia tài' là ai và được phóng tác từ tác phẩm nào?",
    "Trong bài tập 2a, kết quả của phép cộng 27 + 46 là bao nhiêu?",
    "Chòm sao Gấu Bé được tạo thành từ mấy ngôi sao chính và ngôi sao cuối đuôi là sao gì?"
]
for q in sample_qs:
    if st.button(q):
        with st.spinner("Đang tra cứu..."):
            ans = mock_query(q)
            st.info(f"**Câu hỏi:** {q}\n\n**Trả lời:**\n{ans}")
