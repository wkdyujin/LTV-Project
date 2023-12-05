import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px


st.header("1. EDA")
option = st.selectbox(
    'EDA Key 선택',
    ('성별', '연령', '제품'))

# dummy df
import random
from datetime import datetime

# Generating random data
data = []
for i in range(20):
    date = datetime(2022, random.randint(1, 12), random.randint(1, 28))
    invoice_no = i + 1
    product_id = random.randint(100, 999)
    product_name = random.choice(['A', 'B', 'C'])
    quantity = random.randint(1, 10)
    unit_price = round(random.uniform(10, 100), 2)
    customer_id = random.randint(1000, 9999)
    customer_sex = random.choice(['Male', 'Female'])
    customer_age = random.randint(18, 70)

    data.append([date, invoice_no, product_id, product_name, quantity, unit_price, customer_id, customer_sex, customer_age])

# Creating DataFrame
columns = ['InvoiceDate', 'InvoiceNo', 'ProductID', 'ProductName', 'Quantity', 'UnitPrice', 'CustomerID', 'CustomerSex', 'CustomerAge']
df = pd.DataFrame(data, columns=columns)

if option == '성별':
    # 전체 매출 , 제품별 구매량
    col1, col2, = st.columns(2)
    with col1:
        st.write("각 성별의 전체 매출")
        fig1_sex = px.pie(df, values='UnitPrice', names='CustomerSex') # 성별 별 매출
        fig1_sex.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig1_sex, use_container_width=True)


    with col2:
        st.write("제품별 판매량")
        fig2_sex = px.bar(df, x='ProductName', y='Quantity', color='CustomerSex', barmode='group') # 성별 별 제품
        st.plotly_chart(fig2_sex, use_container_width=True)


    
elif option == '연령':
    # 전체 매출 , 제품별 구매량
    col1, col2, = st.columns(2)
    with col1:
        st.write("각 연령의 전체 매출")
        fig1_age = px.pie(df, values='UnitPrice', names='CustomerAge') # 나이 별 매출
        fig1_age.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig1_age, use_container_width=True)


    with col2:
        st.write("연령별 판매량")
        fig2_age = px.bar(df, x='ProductName', y='Quantity', color='CustomerAge', barmode='group', color_continuous_scale='Agsunset') # 제품 별 제품
        st.plotly_chart(fig2_age, use_container_width=True)
    
else:
    col1, col2, = st.columns(2)
    with col1:
        st.write("제품 별 전체 매출")
        fig1_product = px.pie(df, values='UnitPrice', names='ProductName') # 제품 별 매출
        fig1_product.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig1_product, use_container_width=True)


    with col2:
        st.write("제품별 판매량")
        fig2_product = px.bar(df, x='ProductName', y='Quantity', color='ProductName', barmode='group') # 제품 별 제품
        st.plotly_chart(fig2_product, use_container_width=True)
    

st.header("2. LTU")

st.header("3. Regression")

st.header("4. sheet")
st.dataframe(df)