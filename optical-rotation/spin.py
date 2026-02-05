import streamlit as st
import time

with st.spinner("計算中です...しばらくお待ちください"):
    time.sleep(5)

st.success("計算が完了しました！")
