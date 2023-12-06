import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
from io import BytesIO


df = pd.read_csv('./data/sample2.csv', index_col = 0)
df_og = df.copy()
df_output = pd.read_csv('./data/Woori_Output.csv', index_col = 0)

def age_categorize(age):
    age = (age//10) * 10
    return age

df['CustomerAge'] = df['CustomerAge'].astype('Int64')
df['AgeCategory'] = df['CustomerAge'].apply(age_categorize).astype('category') # 연령대로 범주화
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Month'] = df['InvoiceDate'].dt.month_name()
df['TotalSales'] = df['Quantity'] * df['UnitPrice']

st.header("1. EDA")
option = st.selectbox(
    'EDA에 활용될 Key를 선택해 주세요.',
    ('성별', '연령', '제품'))

if option == '성별':
    col1, col2, = st.columns(2)
    with col1:
        st.write("전체 매출 중 각 성별의 비율")
        # st.markdown("<p style='text-align: center;'>전체 매출 중 각 성별의 비율</p>", unsafe_allow_html=True)
        fig1_sex = px.pie(df, values='UnitPrice', names='CustomerSex') # 2성별 별 매출
        fig1_sex.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig1_sex, use_container_width=True)

    with col2:
        st.write("각 성별의 제품 별 구매량")
        # st.markdown("<p style='text-align: center;'>제품 별 판매량</p>", unsafe_allow_html=True)
        fig2_sex = px.bar(df, x='ProductID', y='Quantity', color='CustomerSex', barmode='group',) # 성별 별 제품
        st.plotly_chart(fig2_sex, use_container_width=True)

    st.write("각 성별의 월 매출")
    fig2_sex = px.bar(df, x='Month', y='TotalSales', color='CustomerSex', barmode='group') # 성별 별 제품
    st.plotly_chart(fig2_sex, use_container_width=True)


    
elif option == '연령':
    # 전체 매출 , 제품별 구매량
    col1, col2, = st.columns(2)
    with col1:
        st.write("전체 매출 중 각 연령대의 비율")
        fig1_age = px.pie(df, values='UnitPrice', names='AgeCategory') # 나이 별 매출
        fig1_age.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig1_age, use_container_width=True)

    with col2:
        st.write("각 연령대의 제품 별 구매량")
        df_sum_quantity = df.groupby(['ProductID', 'AgeCategory']).sum(numeric_only=True).reset_index()
        fig2_age = px.bar(df_sum_quantity, x='ProductID', y='Quantity', color='AgeCategory', barmode='group', color_continuous_scale='Agsunset') # 제품 별 제품
        st.plotly_chart(fig2_age, use_container_width=True)

    st.write("각 연령의 월 매출")
    fig = px.bar(df, x='Month', y='TotalSales', color='AgeCategory',
                labels={'TotalSales': 'Sales'},
                color_continuous_scale='Agsunset',
                )
    st.plotly_chart(fig, use_container_width=True)
    
else:
    col1, col2, = st.columns(2)
    with col1:
        st.write("전체 매출 중 각 제품의 비율")
        fig1_product = px.pie(df, values='UnitPrice', names='ProductID') # 제품 별 매출
        fig1_product.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig1_product, use_container_width=True)


    with col2:
        st.write("각 제품의 전체 판매량")
        fig2_product = px.bar(df, x='ProductID', y='Quantity', color='ProductID', barmode='group') # 제품 별 제품
        st.plotly_chart(fig2_product, use_container_width=True)
    
    # Create a stacked bar graph using plotly.express
    st.write("각 제품의 월 매출")
    fig = px.bar(df, x='Month', y='TotalSales', color='ProductID',
                labels={'TotalSales': 'Sales'}
                )
    st.plotly_chart(fig, use_container_width=True)

st.header("2. sheet")
input_excel_data = BytesIO()      
df_og.to_excel(input_excel_data)
st.download_button("엑셀 다운로드", 
        input_excel_data, file_name='input.xlsx')
st.dataframe(df_og)

st.header("3. LTV")
output_excel_data = BytesIO()
df_output.to_excel(output_excel_data)
st.download_button("엑셀 다운로드", 
        output_excel_data, file_name='output.xlsx')
st.dataframe(df_output)

st.header("4. Regression")