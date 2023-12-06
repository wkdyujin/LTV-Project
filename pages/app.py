import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
from io import BytesIO
from functools import reduce
import matplotlib.pyplot as plt

from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse

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
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
label_list = { "CustomerSex": "성별",  "Month": "월", "TotalSales": "매출액", "ProductID": "제품명", "Quantity": "판매량", "AgeCategory": "연령대"}

st.header("1. EDA")
option = st.selectbox(
    'EDA에 활용될 Key를 선택해 주세요.',
    ('성별', '연령', '제품'))

if option == '성별':
    col1, col2, = st.columns(2)
    with col1:
        st.write("전체 매출 중 각 성별의 비율")
        fig1_sex = px.pie(df, values='UnitPrice', names='CustomerSex')
        fig1_sex.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig1_sex, use_container_width=True)

    with col2:
        st.write("각 성별의 제품 별 구매량")
        fig2_sex = px.bar(df, x='ProductID', y='Quantity', color='CustomerSex', barmode='group',
                          labels=label_list, category_orders={ "ProductID": ["A", "B", "C"] },)
        st.plotly_chart(fig2_sex, use_container_width=True)

    st.write("각 성별의 월 매출")
    fig3_sex = px.bar(df, x='Month', y='TotalSales', color='CustomerSex', barmode='group',
                          labels=label_list, category_orders={ "Month": months },)
    st.plotly_chart(fig3_sex, use_container_width=True)
    
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
        fig2_age = px.bar(df_sum_quantity, x='ProductID', y='Quantity', color='AgeCategory', barmode='group', color_continuous_scale='Agsunset', labels=label_list) # 제품 별 제품
        st.plotly_chart(fig2_age, use_container_width=True)

    st.write("각 연령의 월 매출")
    fig = px.bar(df, x='Month', y='TotalSales', color='AgeCategory',
                labels=label_list, color_continuous_scale='Agsunset',
                category_orders={ "Month": months },
                )
    st.plotly_chart(fig, use_container_width=True)
    
else:
    col1, col2, = st.columns(2)
    with col1:
        st.write("전체 매출 중 각 제품의 비율")
        fig1_product = px.pie(df, values='UnitPrice', names='ProductID', labels=label_list) # 제품 별 매출
        fig1_product.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig1_product, use_container_width=True)


    with col2:
        st.write("각 제품의 전체 판매량")
        fig2_product = px.bar(df, x='ProductID', y='Quantity', color='ProductID', barmode='group', labels=label_list) # 제품 별 제품
        st.plotly_chart(fig2_product, use_container_width=True)
    
    st.write("각 제품의 월 매출")
    fig = px.bar(df, x='Month', y='TotalSales', color='ProductID',
                labels=label_list, category_orders={ "Month": months })
    st.plotly_chart(fig, use_container_width=True)

st.header("2. 입력 데이터")
st.text("아래는 입력 데이터(Woori_Input.csv)입니다. 1000 X 8 크기로 구성되어 있습니다.")
st.dataframe(df_og)

st.header("3. 고객 생애 가치")
vvvip_id = df_output[df_output['monetary_value'] == max(df_output['monetary_value'])].index[0]
vvvip = df_output[df_output.index==vvvip_id]
st.text(f"""분석에 따르면, {vvvip_id}가 VVVIP입니다.
- 평균 구매 금액은 {round(vvvip['monetary_value'].values[0], 2)}입니다.
- {int(vvvip['T'].values[0])}일 동안 {int(vvvip['frequency'].values[0])}번 구매했습니다.
- 최근 구매는 {int(vvvip['T'].values[0]- vvvip['recency'].values[0])}일 전에 구매했습니다.
- 예상 구매 횟수는 {int(round(vvvip['predicted_puchases'].values[0], 0))}번 입니다.
- 예상 평균 구매 금액은 {round(vvvip['predicted_monetary_value'].values[0], 4)}입니다.""")
st.dataframe(df_output)

st.header("4. 시계열")

# 데이터프레임 생성
def date_groups(input_data: pd.DataFrame, product: str, unitprice: int):
    # [input_data에 'data'가 들어올 예정]
    
    # InvoiceDate 변수 datetime 으로 바꾸기
    input_data["InvoiceDate"] = pd.to_datetime(input_data["InvoiceDate"]).dt.date

    # Product별 데이터프레임 각각 생성
    product_data = input_data.loc[input_data["ProductID"]==product, :]
    product_data.sort_values("InvoiceDate")

    # Date를 기준으로 groupby -> 일별 Quantity (DailyQuantity) 생성
    product_date_group = product_data.groupby("InvoiceDate", as_index = False)["Quantity"].sum()
    product_date_group.columns = ["InvoiceDate", "DailyQuantity"]
    product_date_group["DailySales"] = product_date_group["DailyQuantity"] * unitprice

    # product_date_group1 : "InvoiceDate"와 "DailySales"만 있는 데이터프레임
    product_date_group1 = product_date_group.loc[:,["InvoiceDate", "DailySales"]]
    product_date_group1.rename(columns = {"DailySales": product}, inplace = True)

    return product_date_group1


# VAR 시계열 예측 함수
def VAR_forecast(TimeSeries_df: pd.DataFrame):
    var_data = TimeSeries_df[['A', 'B', 'C']]

    # VAR 모델 생성
    model_var = VAR(var_data)

    # 모델 훈련
    model_var_fitted = model_var.fit()

    # 다음 30일 예측
    forecast_var = model_var_fitted.forecast(model_var_fitted.endog, steps=30)
    return var_data, forecast_var


# 회귀 모형 만들기 : 예측 - x축 : month, y축은 매출액
select = st.selectbox("어떤 카테고리 별로 확인할 지 선택하세요.", ["제품", "연령", "성별"])
st.subheader(f"{select}별 매출액 예상 그래프")

if select == "제품": 
    # 제품별 시계열 그래프
    A_data = date_groups(df, "A", 50)
    B_data = date_groups(df, "B", 100)
    C_data = date_groups(df, "C", 30)

    # 시계열 데이터 분석을 위한 데이터 프레임 생성
    # Left join
    TimeSeries_data = reduce(lambda x ,y: pd.merge(x, y, on = "InvoiceDate", how = 'left'), [A_data, B_data, C_data])

    # Nan -> 0 : 변수가 일별 매출액(DailySales)이기 때문에 InvoiceDate가 없다면(NaN), 해당 날짜에는 해당 제품의 수요가 없었다는 것이므로 일별 매출액을 0원으로 변경해도 된다고 판단.
    TimeSeries_data = TimeSeries_data.fillna(0)

    # 인덱스 설정
    index_ex = list(TimeSeries_data.InvoiceDate)
    TimeSeries_data.index = pd.DatetimeIndex(index_ex)
    TimeSeries_data = TimeSeries_data.drop("InvoiceDate", axis = 1)

    #st.write(TimeSeries_data)

    # 그래프 그리기
    var_data, forecast_var = VAR_forecast(TimeSeries_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, product in enumerate(['A', 'B', 'C']):
        ax.plot(TimeSeries_data.index, var_data[product], label=f'{product} Actual', linestyle='dashed')
        ax.plot(pd.date_range(start=TimeSeries_data.index[-1], periods=31, freq='D')[1:], forecast_var[:, i], label=f'{product} Forecast')

    ax.set_title('VAR Forecast for Products A, B, C')
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily Sales')
    ax.legend()
    st.pyplot(fig)