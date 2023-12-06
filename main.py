import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

try :
    df_Input = pd.read_csv('./data/sample2.csv')
except :
    df_Input = pd.read_csv('data/sample2.csv')

try :
    df_Output = pd.read_csv('./data/Woori_Output.csv')
except :
    df_Output = pd.read_csv('data/Woori_Output.csv')


st.title("우리FISA AI엔지니어링 프로젝트-1")
version_info = "<p style='text-align: right;'>2023-12-06, v1.0.0</p>"
st.markdown(version_info, unsafe_allow_html=True)



# 이미지를 화면에 표시
# st.image(image_path, caption='Your Image Caption', use_column_width=True)
st.image("https://i.imgur.com/mCQTego.png")

st.header("0. 개요")
st.image("https://i.imgur.com/hytGGJj.png")

st.header("1. 분석")

st.subheader("1-1. 탐색적 데이터 분석(EDA)")
st.markdown('입력 데이터 변수 간의 통계적 관계를 분석하기 위해, "성별", "연령", "제품"에 따라 구성 비율을 파이 그래프로, 판매 수량, 월 별 매출 총액을 막대 그래프로 각각 시각화 하였습니다.')


st.subheader("1-2. 회귀 분석 : 자기회귀벡터 시계열 분석(VAR)")
st.markdown("VAR 시계열 분석이란 예측할 변수의 과거 데이터 뿐만 아니라 여러 변수들 사이의 의존성을 고려하여 예측을 제공하는 분석입니다. 독립 변수와 종속 변수를 각각 주문일자, 제품별 매출액(단가 * 일별 판매량)으로 가정함으로써 제품별 일별 매출액 사이의 관계를 반영하였고, 30일 이후의 시점까지 제품별 매출액을 예측하여 그래프로 표현하였습니다.")
st.markdown("그러나, 입력된 데이터는 정상성(stationary)이 확보되지 않아, 본 분석은 통계적으로 유의미하지 않았으며, 차분, 로그 변환을 시도하였음에도 ADF(정상성검정)의 유의수준이 0.05를 초과하였고, 따라서 정상성을 갖지 않는다는 결과를 얻었습니다. 발표 이후, 추후 보완작업으로 데이터 전처리 및 다양한 통계 분석을 시도할 예정입니다.")

st.subheader("1-3. 고객생애가치 분석(LTV)")
st.markdown("LTV는 'Life-Time Value'의 줄임말로, 고객이 평생 동안 기업에게 어느 정도의 금전적 가치를 가져다 주는지를 정량화한 지표를 의미합니다. 이 값은 (예상 구매 횟수) X (예상 평균 수익)으로 계산되며, (예상 구매 횟수)는 BG/NBD(Beta, Geometic / Negative Binomial) 모형을 따르고, 예상 평균 수익은 Gamma-Gamma 모형에 의해 산출됩니다. 이를 위해, R (Recency), F (Frequency), T (Time), M (Monetary Value)의 정보가 필요합니다.")

st.markdown("이 지표는 유료 웹 서비스를 제공하는 업체나 카드 회사, 게임 회사에서 많이 사용되어 왔으나, 최근 많은 시장(거래)가 데이터화가 되고 있기 때문에 더욱 많은 기업에서 마케팅 지표로 활용하고 있습니다.")
st.link_button("예를 들면? 넥슨 인텔리전스 랩스", "https://www.intelligencelabs.tech/7430a289-22e5-4967-b3b8-03a8644b189f")


st.header("2. 데이터")
st.subheader("2-1. 입력 데이터(Woori_Input.csv)")
st.markdown('''아래는 입력 데이터(Woori_Input.csv)에 대한 정보입니다.
            :orange[1,000]개의 데이터 주문 로그 데이터를 사용했으며, 제품은 A, B, C 등 총 :orange[3]개, 고객은 :orange[100]명, 주문 시간은 :orange[2022.01.01]부터 :orange[2022.12.31]로 가정하였습니다.''')
table_col1 = ["InvoiceDate","InvoiceNo","ProductID","Quantity","UnitPrice","CustomerSex","CustomerAge","CustomerID"]
table_col2 = ["Datetime","Integer","String","Integer","Integer","String","Integer","String"]
table_col3 = ["주문일자","주문번호","제품ID","수량","단가","고객성별","고객나이","고객ID"]
table_info1 = pd.DataFrame({
    'Name': table_col1,
    'Type': table_col2,
    'Description': table_col3
})
st.table(data=table_info1)
dwnbtn_input = st.download_button(
        label="입력 데이터 다운로드",
        data=df_Input.to_csv(),
        file_name='Woori_Input.csv',
        key='input',
        mime='text/csv'
        )

st.subheader("2-2. 출력 데이터(Woori_Output.csv)")
st.markdown('''아래는 입력 데이터(Woori_Output.csv)에 대한 정보입니다.  각 고객의 생애 가치에 대한 데이터입니다.''')    
table_col4 = ["CustomerID","frequency","recency","T","monetary_value","LTV","predicted_puchases","predicted_monetary_value"]
table_col5 = ["String","Integer","Integer","Double","Double","Double","Double","Double"]
table_col6 = ["고객ID","고객별 구매 일수","최초구매부터 마지막구매까지의 시간","최초구매부터 집계일까지의 시간","평균 구매 금액","고객 생애 가치","예상 구매 횟수","예상 평균 구매 금액"]
table_info2 = pd.DataFrame({
    'Name': table_col4,
    'Type': table_col5,
    'Description': table_col6
})
st.table(data=table_info2)
dwnbtn_output = st.download_button(
        label="출력 데이터 다운로드",
        data=df_Output.to_csv(),
        file_name='Woori_Input.csv',
        key='output',
        mime='text/csv'
        )


st.markdown('\n\n\n')
st.markdown('''[1] 사용한 데이터는 특정 기업으로부터 입수한 데이터가 아닌 학습 목적의 :red[가상의 데이터] 입니다.''')
st.markdown('''[2] LTV분석은 다음 자료를 참고하였습니다. (URL : https://playinpap.github.io/ltv-practice/)''')

