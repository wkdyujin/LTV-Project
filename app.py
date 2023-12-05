import streamlit as st
import pandas as pd


st.header("1. EDA")
option = st.selectbox(
    'EDA Key 선택',
    ('성별', '연령', '제품'))

if option == '성별':
    st.write('성별로 EDA')
    
elif option == '연령':
    st.write('연령으로 EDA')
    
else:
    st.write('제품으로 EDA')
    

st.header("2. LTU")

st.header("3. Regression")

st.header("4. sheet")