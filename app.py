import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px


df = pd.read_csv('./data/sample2.csv', index_col = 0)
df_og = df.copy()

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
    # 전체 매출 , 제품별 구매량
    col1, col2, = st.columns(2)
    with col1:
        st.write("각 성별 전체 매출")
        fig1_sex = px.pie(df, values='UnitPrice', names='CustomerSex') # 2성별 별 매출
        fig1_sex.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig1_sex, use_container_width=True)

    with col2:
        st.write("제품 별 판매량")
        fig2_sex = px.bar(df, x='ProductID', y='Quantity', color='CustomerSex', barmode='group') # 성별 별 제품
        st.plotly_chart(fig2_sex, use_container_width=True)

    st.write("각 성별의 월별 전체 매출")
    fig2_sex = px.bar(df, x='Month', y='TotalSales', color='CustomerSex', barmode='group') # 성별 별 제품
    st.plotly_chart(fig2_sex, use_container_width=True)


    
elif option == '연령':
    # 전체 매출 , 제품별 구매량
    col1, col2, = st.columns(2)
    with col1:
        st.write("각 연령대 전체 매출")
        fig1_age = px.pie(df, values='UnitPrice', names='AgeCategory') # 나이 별 매출
        fig1_age.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig1_age, use_container_width=True)

    with col2:
        st.write("연령대 별 판매량")
        df_sum_quantity = df.groupby(['ProductID', 'AgeCategory']).sum(numeric_only=True).reset_index()
        fig2_age = px.bar(df_sum_quantity, x='ProductID', y='Quantity', color='AgeCategory', barmode='group', color_continuous_scale='Agsunset') # 제품 별 제품
        st.plotly_chart(fig2_age, use_container_width=True)

    st.write("각 연령별 월별 전체 매출")
    fig = px.bar(df, x='Month', y='TotalSales', color='AgeCategory',
                labels={'TotalSales': 'Sales'},
                color_continuous_scale='Agsunset',
                )
    st.plotly_chart(fig, use_container_width=True)
    
else:
    col1, col2, = st.columns(2)
    with col1:
        st.write("각 제품 전체 매출")
        fig1_product = px.pie(df, values='UnitPrice', names='ProductID') # 제품 별 매출
        fig1_product.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig1_product, use_container_width=True)


    with col2:
        st.write("제품 별 판매량")
        fig2_product = px.bar(df, x='ProductID', y='Quantity', color='ProductID', barmode='group') # 제품 별 제품
        st.plotly_chart(fig2_product, use_container_width=True)
    
    # Create a stacked bar graph using plotly.express
    fig = px.bar(df, x='Month', y='TotalSales', color='ProductID',
                title='A, B, C Sales by Month',
                labels={'TotalSales': 'Sales'},
                )
    st.plotly_chart(fig, use_container_width=True)

st.header("2. sheet")
st.dataframe(df_og)

st.header("3. LTV")

st.header("4. Regression")