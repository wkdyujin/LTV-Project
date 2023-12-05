import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px


df = pd.read_csv('./data/sample2.csv')
df_og = df.copy()

def age_categorize(age):
    age = (age//10) * 10
    return age

df['CustomerAge'] = df['CustomerAge'].astype('Int64')
df['AgeCategory'] = df['CustomerAge'].apply(age_categorize) # 연령대로 범주화


st.header("1. EDA")
option = st.selectbox(
    'EDA Key 선택',
    ('성별', '연령', '제품'))

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
        fig2_sex = px.bar(df, x='ProductID', y='Quantity', color='CustomerSex', barmode='group') # 성별 별 제품
        st.plotly_chart(fig2_sex, use_container_width=True)


    
elif option == '연령':
    # 전체 매출 , 제품별 구매량
    col1, col2, = st.columns(2)
    with col1:
        st.write("각 연령의 전체 매출")
        fig1_age = px.pie(df, values='UnitPrice', names='AgeCategory') # 나이 별 매출
        fig1_age.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig1_age, use_container_width=True)


    with col2:
        st.write("연령별 판매량")
        # TODO: 수정
        fig2_age = px.bar(df, x='ProductID', y='Quantity', color='AgeCategory', barmode='group', color_continuous_scale='Agsunset') # 제품 별 제품
        st.plotly_chart(fig2_age, use_container_width=True)
    
else:
    col1, col2, = st.columns(2)
    with col1:
        st.write("제품 별 전체 매출")
        fig1_product = px.pie(df, values='UnitPrice', names='ProductID') # 제품 별 매출
        fig1_product.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig1_product, use_container_width=True)


    with col2:
        st.write("제품별 판매량")
        fig2_product = px.bar(df, x='ProductID', y='Quantity', color='ProductID', barmode='group') # 제품 별 제품
        st.plotly_chart(fig2_product, use_container_width=True)
    
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Month'] = df['InvoiceDate'].dt.month_name()
    df['TotalSales'] = df['Quantity'] * df['UnitPrice']
    # Create a stacked bar graph using plotly.express
    fig = px.bar(df, x='Month', y='TotalSales', color='ProductID',
                title='A, B, C Sales by Month',
                labels={'TotalSales': 'Sales'},
                color_discrete_map={'A': 'lightblue', 'B': 'orange', 'C': 'green'},
                )
    st.plotly_chart(fig, use_container_width=True)

st.header("2. LTU")

st.header("3. Regression")

st.header("4. sheet")
st.dataframe(df_og)