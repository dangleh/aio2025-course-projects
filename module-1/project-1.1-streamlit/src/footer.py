import streamlit as st

def show_footer():
    footer_html = """
    <div style='text-align: center; margin-top: 50px; font-size:12px; color: grey;'>
        © 2025 lehaidang2601@gmail.com. All rights reserved.
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)