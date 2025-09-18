"""
Streamlit Tutorial Application
Hướng dẫn cơ bản về Streamlit với ứng dụng tính giai thừa và phân tích điểm số
"""

import streamlit as st
import pandas as pd
import numpy as np
from math import factorial

def main():
    st.set_page_config(
        page_title="Streamlit Tutorial",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Streamlit Tutorial")
    st.markdown("Hướng dẫn cơ bản về Streamlit với các ứng dụng mẫu")
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Chọn ứng dụng:",
        ["Tính giai thừa", "Phân tích điểm số", "Hướng dẫn Streamlit"]
    )
    
    if page == "Tính giai thừa":
        factorial_app()
    elif page == "Phân tích điểm số":
        grade_analysis_app()
    else:
        streamlit_tutorial()

def factorial_app():
    st.header("🧮 Ứng dụng tính giai thừa")
    
    col1, col2 = st.columns(2)
    
    with col1:
        number = st.number_input(
            "Nhập số cần tính giai thừa:",
            min_value=0,
            max_value=100,
            value=5,
            step=1
        )
        
        if st.button("Tính giai thừa"):
            result = factorial(number)
            st.success(f"Giai thừa của {number} là: {result:,}")
    
    with col2:
        st.subheader("Giải thích")
        st.markdown("""
        **Giai thừa** của một số nguyên dương n được định nghĩa là:
        - n! = n × (n-1) × (n-2) × ... × 2 × 1
        - 0! = 1
        
        Ví dụ: 5! = 5 × 4 × 3 × 2 × 1 = 120
        """)

def grade_analysis_app():
    st.header("📈 Phân tích điểm số")
    
    # Sample data
    np.random.seed(42)
    data = {
        'Học sinh': [f'Học sinh {i+1}' for i in range(20)],
        'Toán': np.random.normal(7.5, 1.5, 20).round(1),
        'Lý': np.random.normal(7.0, 1.8, 20).round(1),
        'Hóa': np.random.normal(6.8, 1.6, 20).round(1),
        'Sinh': np.random.normal(7.2, 1.4, 20).round(1)
    }
    
    df = pd.DataFrame(data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Bảng điểm")
        st.dataframe(df, use_container_width=True)
    
    with col2:
        st.subheader("📈 Thống kê")
        
        # Calculate statistics
        stats = df.select_dtypes(include=[np.number]).describe()
        st.dataframe(stats)
        
        # Average by subject
        st.subheader("Điểm trung bình theo môn")
        avg_scores = df.select_dtypes(include=[np.number]).mean()
        st.bar_chart(avg_scores)

def streamlit_tutorial():
    st.header("📚 Hướng dẫn Streamlit")
    
    st.markdown("""
    ## Các thành phần cơ bản của Streamlit
    
    ### 1. Text và Markdown
    """)
    
    st.write("Đây là text thường")
    st.markdown("**Đây là markdown**")
    st.code("print('Hello Streamlit!')")
    
    st.markdown("### 2. Widgets")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.button("Button")
        st.checkbox("Checkbox")
        st.radio("Radio", ["Option 1", "Option 2"])
    
    with col2:
        st.selectbox("Selectbox", ["A", "B", "C"])
        st.multiselect("Multiselect", ["X", "Y", "Z"])
        st.slider("Slider", 0, 100, 50)
    
    with col3:
        st.text_input("Text input")
        st.number_input("Number input", 0, 100, 25)
        st.text_area("Text area")
    
    st.markdown("### 3. Data Display")
    
    # Sample chart data
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['A', 'B', 'C']
    )
    
    st.line_chart(chart_data)
    st.area_chart(chart_data)

if __name__ == "__main__":
    main()
