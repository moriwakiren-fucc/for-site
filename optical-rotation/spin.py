import streamlit as st
import time

with st.status("計算を開始します...", expanded=True) as status:
    for i in range(5):
        st.write(f"{i+1} 秒目の処理中...")
        time.sleep(1)
    status.update(label="計算が完了しました ✅", state="complete")
