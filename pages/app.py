import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
from io import BytesIO
from functools import reduce
import matplotlib.pyplot as plt

from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse

st.set_page_config(layout="wide")

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
months = ["January", "February", "March", "April", "May", "June",\
           "July", "August", "September", "October", "November", "December"]
label_list = { "CustomerSex": "성별",  "Month": "월", "TotalSales": "매출액", \
              "ProductID": "제품명", "Quantity": "판매량", "AgeCategory": "연령대"}

months_korean = ["1월", "2월", "3월", "4월", "5월", "6월",
                  "7월", "8월", "9월", "10월", "11월", "12월"]
month_translation = dict(zip(months, months_korean))
df['Month'] = df['Month'].map(month_translation)
months = ["1월", "2월", "3월", "4월", "5월", "6월",
                  "7월", "8월", "9월", "10월", "11월", "12월"]

AgeCategory = ['10', '20', '30', '40', '50', '60']

CustomerSex = ['Male', 'Female']
CustomerSex_korean = ['남성', '여성']
CustomerSex_translation = dict(zip(CustomerSex, CustomerSex_korean))
df['CustomerSex'] = df['CustomerSex'].map(CustomerSex_translation)

st.header("1. EDA")
option = st.selectbox(
    'EDA에 활용될 Key를 선택해 주세요.',
    ('성별', '연령', '제품'))

if option == '성별':
    col1, col2, = st.columns(2)
    with col1:
        st.subheader("1-1. 각 성별의 비율")
        st.markdown("전체 기간 동안 각 성별이 매출액에서 차지하는 비율을 나타냅니다.")
        fig1_sex = px.pie(df, values='UnitPrice', names='CustomerSex')
        fig1_sex.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig1_sex, use_container_width=True)

    with col2:
        st.subheader("1-2. 각 성별의 제품별 구매량")
        st.markdown("전체 기간 동안 각 성별이 구매한 제품 별 수량을 나타냅니다.")
        fig2_sex = px.bar(df, x='ProductID', y='Quantity', color='CustomerSex', barmode='group',
                          labels=label_list, category_orders={ "ProductID": ["A", "B", "C"] },)
        st.plotly_chart(fig2_sex, use_container_width=True)

    st.subheader("1-3. 각 성별의 월 매출")
    st.markdown("각 성별의 월 별 매출을 나타냅니다.")
    fig3_sex = px.bar(df, x='Month', y='TotalSales', color='CustomerSex', barmode='group',
                          labels=label_list, category_orders={ "Month": months },)
    st.plotly_chart(fig3_sex, use_container_width=True)
    
elif option == '연령':
    # 전체 매출 , 제품별 구매량
    col1, col2, = st.columns(2)
    with col1:
        st.subheader("1-1. 각 연령대의 비율")
        st.markdown("전체 기간 동안 각 연령대가 매출액에서 차지하는 비율을 나타냅니다.")
        fig1_age = px.pie(df, values='UnitPrice', names='AgeCategory') # 나이 별 매출
        fig1_age.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig1_age, use_container_width=True)

    with col2:
        st.subheader("1-2. 각 연령대의 제품별 판매량")
        st.markdown("전체 기간 동안 각 연령대가 구매한 제품 별 수량을 나타냅니다.")
        df_sum_quantity = df.groupby(['ProductID', 'AgeCategory']).sum(numeric_only=True).reset_index()
        fig2_age = px.bar(df_sum_quantity, x='ProductID', y='Quantity', color='AgeCategory', barmode='group', color_continuous_scale='Agsunset') # 제품 별 제품
        st.plotly_chart(fig2_age, use_container_width=True)

    st.subheader("1-3. 각 연령의 월 매출")
    st.markdown("각 연령의 월 별 매출을 나타냅니다.")
    fig = px.bar(df, x='Month', y='TotalSales', color='AgeCategory',
                labels={'TotalSales': 'Sales'},
                color_continuous_scale='Agsunset',
                )
    st.plotly_chart(fig, use_container_width=True)
    
else:
    col1, col2, = st.columns(2)
    with col1:
        st.subheader("1-1. 각 제품의 비율")
        st.markdown("전체 기간 동안 각 제품이 매출액에서 차지하는 비율을 나타냅니다.")
        fig1_product = px.pie(df, values='UnitPrice', names='ProductID', labels=label_list) # 제품 별 매출
        fig1_product.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig1_product, use_container_width=True)

    with col2:
        st.subheader("1-2. 각 제품의 전체 판매량")
        st.markdown("전체 기간 동안 각 제품 별 전체 판매량을 나타냅니다.")
        fig2_product = px.bar(df, x='ProductID', y='Quantity', color='ProductID', barmode='group', labels=label_list) # 제품 별 제품
        st.plotly_chart(fig2_product, use_container_width=True)
    
    st.subheader("1-3. 각 제품의 월 매출")
    st.markdown("각 제품의 월 별 매출을 나타냅니다.")
    fig = px.bar(df, x='Month', y='TotalSales', color='ProductID',
                labels={'TotalSales': 'Sales'}
                )
    st.plotly_chart(fig, use_container_width=True)

st.header("2. 입력데이터")
st.dataframe(df_og)

st.header("3. 고객 생애 가치")
vvvip_id = df_output[df_output['LTV'] == max(df_output['LTV'])].index[0]
vvvip = df_output[df_output.index==vvvip_id]
# st.text(f"분석에 따르면, {vvvip_id}가 VVVIP입니다.")
st.markdown(f"분석에 따르면, <span style='color: orange;'><b>{vvvip_id}</b>가 <b>VVVIP</b>입니다.</span>", unsafe_allow_html=True)

ans_customer_id = st.text_input("분석할 고객 ID를 입력해 주세요.")
if (ans_customer_id != ''):
    if (ans_customer_id not in df_output.index):
        st.text("해당 ID는 존재하지 않습니다.")
    else:
        cur_customer = df_output[df_output.index==ans_customer_id]
        st.text(f"{ans_customer_id}의 분석 결과입니다.")
        st.text(f"""
        - 평균 구매 금액은 {round(cur_customer['monetary_value'].values[0], 2)}입니다.
        - {int(cur_customer['T'].values[0])}일 동안 {int(cur_customer['frequency'].values[0])}번 구매했습니다.
        - 최근 구매는 {int(cur_customer['T'].values[0]- cur_customer['recency'].values[0])}일 전에 구매했습니다.
        - 예상 구매 횟수는 {int(round(cur_customer['predicted_puchases'].values[0], 0))}번 입니다.
        - 예상 평균 구매 금액은 {round(cur_customer['predicted_monetary_value'].values[0], 2)}입니다.""")
st.dataframe(df_output)

st.header("4. 시계열")
# st.write(df)

# 1) 제품별 데이터프레임 생성
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


# 1-1) 제품별 VAR 시계열 예측 함수
def VAR_forecast(TimeSeries_df: pd.DataFrame):
    var_data = TimeSeries_df[['A', 'B', 'C']]

    # VAR 모델 생성
    model_var = VAR(var_data)

    # 모델 훈련
    model_var_fitted = model_var.fit()

    # 다음 30일 예측
    forecast_var = model_var_fitted.forecast(model_var_fitted.endog, steps=30)
    return var_data, forecast_var


# 2) 성별 데이터프레임 생성
# 시계열 데이터 : 성별 DailySales 구하기
def sex_date_groups(input_data: pd.DataFrame, sex: str):
    # [input_data에 'data'가 들어올 예정]
    
    # InvoiceDate 변수 datetime 으로 바꾸기
    input_data["InvoiceDate"] = pd.to_datetime(input_data["InvoiceDate"]).dt.date
    
    # Product별 데이터프레임 각각 생성
    product_data = input_data.loc[input_data["CustomerSex"]== sex, :]
    product_data.sort_values("InvoiceDate")

    # Date를 기준으로 groupby -> 일별 Quantity (DailyQuantity) 생성
    product_date_group = product_data.groupby(["InvoiceDate"], as_index = False)["Quantity"].sum()
    product_date_group.columns = ["InvoiceDate", "DailyQuantity"]

    # Left Join
    sex_left_join = pd.merge(product_date_group, product_data, left_on ="InvoiceDate", right_on = "InvoiceDate", how = "left")
    sex_left_join["DailySales"] = sex_left_join["DailyQuantity"]*sex_left_join["UnitPrice"]

    # for_timeseries : "InvoiceDate"와 "DailySales"만 있는 데이터프레임
    for_timeseries = sex_left_join.loc[:,["InvoiceDate", "DailySales"]]
    for_timeseries.rename(columns = {"DailySales": sex}, inplace = True)

    return sex_left_join, for_timeseries

# 2-1) 성별 VAR 시계열 예측
def Sex_VAR_forecast(TimeSeries_df: pd.DataFrame):
    var_data = TimeSeries_df[['여성', '남성']]
    # VAR 모델 생성
    model_var = VAR(var_data)
    # 모델 훈련
    model_var_fitted = model_var.fit()
    # 다음 30일 예측
    forecast_var = model_var_fitted.forecast(model_var_fitted.endog, steps=30)
    return var_data, forecast_var


# 3) 연령별 데이터 프레임 생성
def Age_date_groups(input_data: pd.DataFrame, ageline):
    # [input_data에 'data'가 들어올 예정]

    # InvoiceDate 변수 datetime 으로 바꾸기
    input_data["InvoiceDate"] = pd.to_datetime(input_data["InvoiceDate"]).dt.date
    
    # AgeLine 파생변수 생성 : 연령대
    input_data["AgeLine"] = (input_data["CustomerAge"]/10).astype(int) * 10
    
    # Product별 데이터프레임 각각 생성
    product_data = input_data.loc[input_data["CustomerAge"]== int(ageline), :]
    product_data.sort_values("InvoiceDate")

    # Date를 기준으로 groupby -> 일별 Quantity (DailyQuantity) 생성
    product_date_group = product_data.groupby(["InvoiceDate"], as_index = False)["Quantity"].sum()
    product_date_group.columns = ["InvoiceDate", "DailyQuantity"]

    # Left Join
    age_left_join = pd.merge(product_date_group, product_data, left_on ="InvoiceDate", right_on = "InvoiceDate", how = "left")
    age_left_join["DailySales"] = age_left_join["DailyQuantity"]*age_left_join["UnitPrice"]

    # for_timeseries : "InvoiceDate"와 "DailySales"만 있는 데이터프레임
    for_timeseries = age_left_join.loc[:,["InvoiceDate", "DailySales"]]
    for_timeseries.rename(columns = {"DailySales": ageline}, inplace = True)

    # product_date_group1 = product_date_group.loc[:,["InvoiceDate", "DailySales"]]
    # product_date_group1.rename(columns = {"DailySales": product}, inplace = True)

    return age_left_join, for_timeseries

# 3-1) 연령별 VAR 시계열 예측
def Age_VAR_forecast(TimeSeries_df: pd.DataFrame):
    var_data = TimeSeries_df[['10', '20', '30', '40', '50']]

    # VAR 모델 생성
    model_var = VAR(var_data)

    # 모델 훈련
    model_var_fitted = model_var.fit()

    # 다음 30일 예측
    forecast_var = model_var_fitted.forecast(model_var_fitted.endog, steps=30)
    return var_data, forecast_var


# UI : 시계열 예측 그래프 출력
select = st.selectbox("어떤 카테고리 별로 확인할 지 선택하세요.", ["제품", "연령", "성별"])
st.subheader(f"{select}별 매출액 예상 그래프")

# 제품별 시계열 예측 그래프
if select == "제품": 
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

# 연령별 시계열 예측 그래프
if select == "연령":
    teen_left_join, teen_time = Age_date_groups(df, '10')
    twen_left_join, twen_time = Age_date_groups(df, '20')
    thir_left_join, thir_time = Age_date_groups(df, '30')
    four_left_join, four_time = Age_date_groups(df, '40')
    fith_left_join, fith_time = Age_date_groups(df, '50')

    # Age_TimeSeries_data
    # Outer join
    Age_TimeSeries_data = pd.concat([teen_time, twen_time, thir_time, four_time, fith_time]).sort_values("InvoiceDate")

    # Nan -> 0 : 변수가 일별 매출액(DailySales)이기 때문에 InvoiceDate가 없다면(NaN), 해당 날짜에는 해당 제품의 수요가 없었다는 것이므로 일별 매출액을 0원으로 변경해도 된다고 판단.
    Age_TimeSeries_data = Age_TimeSeries_data.fillna(0)

    # 인덱스 설정
    index_ex = list(Age_TimeSeries_data.InvoiceDate)
    Age_TimeSeries_data.index = pd.DatetimeIndex(index_ex)
    Age_TimeSeries_data = Age_TimeSeries_data.drop("InvoiceDate", axis = 1)
    # st.write(Age_TimeSeries_data)

    # 그래프 그리기
    age_var_data, age_forecast_var = Age_VAR_forecast(Age_TimeSeries_data)
    # st.write(Age_TimeSeries_data[:1])
    # st.write(age_var_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, age in enumerate(['10', '20', '30', '40', '50']):
        ax.plot(Age_TimeSeries_data.index, age_var_data[age], label=f'{age} Actual', linestyle='dashed')
        ax.plot(pd.date_range(start = Age_TimeSeries_data.index[-1], periods=31, freq='D')[1:], age_forecast_var[:, i], label=f'{age} Forecast')

    ax.set_title('VAR Forecast for Age')
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily Sales')
    ax.legend()
    st.pyplot(fig)


# 성별 시계열 예측 그래프

if select == "성별" :
    Female_left_join, Female_time = sex_date_groups(df, "여성")
    Male_left_join, Male_time = sex_date_groups(df, "남성")
    
    # Sex_TimeSeries_data
    # Left join
    Sex_TimeSeries_data = pd.merge(Female_time, Male_time, on = "InvoiceDate", how = 'left')

    # Nan -> 0 : 변수가 일별 매출액(DailySales)이기 때문에 InvoiceDate가 없다면(NaN), 해당 날짜에는 해당 제품의 수요가 없었다는 것이므로 일별 매출액을 0원으로 변경해도 된다고 판단.
    Sex_TimeSeries_data = Sex_TimeSeries_data.fillna(0)

    # 인덱스 설정
    index_ex = list(Sex_TimeSeries_data.InvoiceDate)
    Sex_TimeSeries_data.index = pd.DatetimeIndex(index_ex)
    Sex_TimeSeries_data = Sex_TimeSeries_data.drop("InvoiceDate", axis = 1)


    # 그래프 그리기
    sex_var_data, sex_forecast_var = Sex_VAR_forecast(Sex_TimeSeries_data)

    sex_english = {"여성": "Female", "남성": "Male"}

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, sex in enumerate(['여성', '남성']):
        ax.plot(Sex_TimeSeries_data.index, sex_var_data[sex], label=f'{sex_english[sex]} Actual', linestyle='dashed')
        ax.plot(pd.date_range(start=Sex_TimeSeries_data.index[-1], periods=31, freq='D')[1:], sex_forecast_var[:, i], label=f'{sex_english[sex]} Forecast')

    ax.set_title('VAR Forecast for Sex')
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily Sales')
    ax.legend(loc = 'upper right')
    st.pyplot(fig)