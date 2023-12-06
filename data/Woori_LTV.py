
#! Import Library
import pandas as pd
import warnings
import os
import numpy as np

from matplotlib import pyplot as plt
from scipy.stats import gamma
from scipy.stats import beta 

from datetime import datetime
from datetime import timedelta

from lifetimes.plotting import *
from lifetimes.utils import *
from lifetimes import BetaGeoFitter
from lifetimes.fitters.gamma_gamma_fitter import GammaGammaFitter

from hyperopt import hp
from hyperopt import fmin
from hyperopt import tpe
from hyperopt import rand
from hyperopt import SparkTrials
from hyperopt import STATUS_OK
from hyperopt import space_eval
from hyperopt import Trials

l2_reg = 0.05
holdout_days = 90

# 최적화 작업을 위해, 아래 3개의 함수를 정의
#! score_model : 실제값과 예측값의 차이에 대한 지표 (MSE / RMSE / MAE)를 계산하는 함수
def score_model(actuals, predicted, metric = 'mse') :
    if type(actuals) == 'list' :
        actuals = np.array(actuals)
    if type(predicted) == 'list' :
        predicted = np.array(predicted)
    
    val = None    
    
    # MSE
    if metric.lower() == 'mse' or metric.lower() == 'rmse' :
        val = np.sum(np.square(actuals - predicted)) / actuals.shape[0]
    
    # RMSE
    elif metric == 'rmse' :
        val = np.sqrt(
            np.sum(np.square(actuals - predicted)) / actuals.shape[0]
            )
        
    # MAE
    elif metric == 'mae' :
        val = np.sum(np.abs(actuals - predicted)) / actuals.shape[0]
        
    return val    

def evaluate_BG_NBD_model(inputs, param) :
    data    = inputs
    l2_reg  = param
    
    # BG/NBD 모형 피팅
    model = BetaGeoFitter(penalizer_coef = l2_reg)
    model.fit(data['frequency_cal'], data['recency_cal'], data['T_cal'])
    
    # 모형 평가
    frequency_actual = data['frequency_holdout']
    frequency_predicted = model.predict(data['duration_holdout'],
                                        data['frequency_cal'],
                                        data['recency_cal'],
                                        data['T_cal']
                                        )   

    mse = score_model(frequency_actual, frequency_predicted, metric = 'mse' )
    return {'loss' : mse, 'status' : STATUS_OK}
    
def evaluate_GG_model(inputs, param) :
    data    = inputs
    l2_reg  = param

    # GammaGamma 모형 피팅
    model = GammaGammaFitter(penalizer_coef = l2_reg)
    model.fit(data['frequency_cal'], data['monetary_value_cal'])

    # 모형 평가
    monetary_actual = data['monetary_value_holdout']
    monetary_predicted = model.conditional_expected_average_profit(
        data['frequency_holdout'],
        data['monetary_value_holdout']
        )
    mse = score_model(monetary_actual, monetary_predicted)
    return {'loss' : mse, 'status' : STATUS_OK}

def find_L2penalty_BG_NBD_model(filtered_df):
    search_space = hp.uniform('l2', 0.0, 1.0)
    algo = tpe.suggest
    trials = Trials()
    inputs = filtered_df
    
    def tmp_evaluate_BG_NBD_model(param) :
        return evaluate_BG_NBD_model(inputs, param)
    
    argmin = fmin(
        fn = tmp_evaluate_BG_NBD_model,     # 목적 함수
        space = search_space,           # 파라미터 공간
        algo = algo,                    # 최적화 알고리즘 : Tree of Parzen Estimators (TPE)
        max_evals = 100,                # iteration
        trials = trials
    )
    
    l2_bgnbd = space_eval(search_space, argmin)
    return l2_bgnbd

def find_L2penalty_GG_model(filtered_df) :
    search_space = hp.uniform('l2', 0.0, 1.0)
    algo = tpe.suggest
    trials = Trials()
    inputs = filtered_df
    
    def tmp_evaluate_GG_model(param) :
        return evaluate_GG_model(inputs, param)
        
    # GammaGamma
    argmin = fmin(
        fn = tmp_evaluate_GG_model,
        space = search_space,
        algo = algo,
        max_evals = 100,
        trials = trials
    )
    
    l2_gg = space_eval(search_space, argmin)
    return l2_gg

def calibration_and_holdout_data_copy(
    transactions,
    customer_id_col,
    datetime_col,
    calibration_period_end,
    observation_period_end=None,
    freq = "D",
    freq_multiplier = 1,
    datetime_format = None,
    monetary_value_col = None,
    include_first_transaction = False,
):

    def to_period(d):
        return d.to_period(freq)

    if observation_period_end is None:
        observation_period_end = transactions[datetime_col].max()

    transaction_cols = [customer_id_col, datetime_col]
    if monetary_value_col:
        transaction_cols.append(monetary_value_col)
    transactions = transactions[transaction_cols].copy()

    transactions[datetime_col] = pd.to_datetime(transactions[datetime_col], format=datetime_format)
    observation_period_end = pd.to_datetime(observation_period_end, format=datetime_format)
    calibration_period_end = pd.to_datetime(calibration_period_end, format=datetime_format)

    # create calibration dataset
    calibration_transactions = transactions.loc[transactions[datetime_col] <= calibration_period_end]
    calibration_summary_data = summary_data_from_transaction_data(
        calibration_transactions,
        customer_id_col,
        datetime_col,
        datetime_format=datetime_format,
        observation_period_end=calibration_period_end,
        freq=freq,
        freq_multiplier=freq_multiplier,
        monetary_value_col=monetary_value_col,
        include_first_transaction=include_first_transaction,
    )
    calibration_summary_data.columns = [c + "_cal" for c in calibration_summary_data.columns]

    # create holdout dataset
    holdout_transactions = transactions.loc[
        (observation_period_end >= transactions[datetime_col]) & (transactions[datetime_col] > calibration_period_end)
    ]

    if holdout_transactions.empty:
        raise ValueError(
            "There is no data available. Check the `observation_period_end` and  `calibration_period_end` and confirm that values in `transactions` occur prior to those dates."
        )

    holdout_transactions[datetime_col] = holdout_transactions[datetime_col].map(to_period)
    holdout_summary_data = (
        holdout_transactions.groupby([customer_id_col, datetime_col], sort=False)
        .agg(lambda r: 1)
        .groupby(level=customer_id_col)
        .agg(["count"])
    )
    holdout_summary_data.columns = ["frequency_holdout"]
    if monetary_value_col:
        holdout_summary_data["monetary_value_holdout"] = holdout_transactions.groupby(customer_id_col)[
            monetary_value_col
        ].mean()

    combined_data = calibration_summary_data.join(holdout_summary_data, how="left")
    combined_data.fillna(0, inplace=True)

    delta_time = (to_period(observation_period_end) - to_period(calibration_period_end)).n
    combined_data["duration_holdout"] = delta_time / freq_multiplier

    return combined_data


def run_total_process():
    #! Import Data
    PATH = os.getcwd().split('\\')
    MAIN_PATH = r''
    for ii in range(len(PATH)) :
        MAIN_PATH += str(PATH[ii] + r'/') 
    
    df = pd.read_csv(MAIN_PATH + r'Woori_invoice.csv')
    # print (df.head(10))

    # InvoiceDate (주문 일자) : Datetime -> Date
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']).dt.date

    # CustomerID : NULL 제외
    df = df[pd.notnull(df['CustomerID'])]

    # Quantity (주문 수량) : 1 이상
    df = df[df['Quantity'] >= 1]

    # Sales (구매 금액) 변수 생성
    df['Sales'] = df['Quantity'] * df['UnitPrice']

    # 고객 번호, 주문 일자, 구매 금액만 남기고 모두 지우기
    cols_of_interest = ['CustomerID', 'InvoiceDate', 'Sales']
    df = df[cols_of_interest]
 

    #! lifetimes 패키지로 RFMT 계산
    current_date = df['InvoiceDate'].max()

    metrics_df = summary_data_from_transaction_data(df,
                                                    customer_id_col = 'CustomerID',
                                                    datetime_col = 'InvoiceDate',
                                                    monetary_value_col = 'Sales',
                                                    observation_period_end = current_date
                                                    )

    # L2 penalty의 계수를 어떻게 입력해야 효율적인지
    # 파리미터를 찾는 과정이 필요 -> Calibration
    # 어떤 L2 penality가 '최적'인지 알기 위해 적당하게 데이터를 나누어 훈련 시키고
    # 테스트를 하는 과정이 필요
  
    calibration_end_date = current_date - timedelta(days = holdout_days)

    metrics_cal_df = calibration_and_holdout_data_copy(df,
                                                customer_id_col = 'CustomerID',
                                                datetime_col = 'InvoiceDate',
                                                calibration_period_end = calibration_end_date,
                                                observation_period_end = current_date,
                                                monetary_value_col = 'Sales')
    # print (metrics_cal_df.head(10))
    # frequency가 0인 row는 제외하기


    # whole_filterd_df
    # L2 페널티를 최적한 후
    # 가장 마지막 LTV를 계산할 때 사용할 데이터
    # calibration / holdout을 나누지 않은 데이터
    whole_filtered_df       = metrics_df[metrics_df.frequency > 0] 

    # filered_df는 L2를 최적화 하기 위해
    # calibration / holdout을 나눈 데이터
    filtered_df             = metrics_cal_df[metrics_cal_df.frequency_cal > 0]
    
    l2_bgnbd = find_L2penalty_BG_NBD_model(filtered_df)
    l2_gg = find_L2penalty_GG_model(filtered_df)
    
    
    #! BG_NBD 모델 피팅
    lifetimes_model = BetaGeoFitter(penalizer_coef = l2_bgnbd)    
    # calibration 데이터의 R, F, T로 모형 피팅
    lifetimes_model.fit(filtered_df['frequency_cal'],
                        filtered_df['recency_cal'],
                        filtered_df['T_cal']
                        )
    
    # holdout 데이터로 모델 평가 : F의 실제값과 예측값의 MSE
    frequency_actual = filtered_df['frequency_holdout']
    print (frequency_actual.index)
    frequency_predicted = lifetimes_model.predict(filtered_df['duration_holdout'],
                                                    filtered_df['frequency_cal'],
                                                    filtered_df['recency_cal'],
                                                    filtered_df['T_cal']
                                                    )

    mse = score_model(frequency_actual, frequency_actual, 'mse')
    
    print('MSE : {0}'.format(mse))
    ## >> MSE : 2.9944204820420146
    ## >> 오차가 +- 3일 정도 됨
    
    #print (lifetimes_model.summary)
    param_r     = lifetimes_model.summary['coef']['r']
    param_alpha = lifetimes_model.summary['coef']['alpha']
    param_a     = lifetimes_model.summary['coef']['a']
    param_b     = lifetimes_model.summary['coef']['b']

    """
    0. 위 산출된 param_r, param_alpha, param_a, param_b은 아래 설명 내
        r, \alpha, a, b에 대응한다.

    1. 고객이 남아 있는 동안, 일정 기간 T 동안의 구매 횟수는 Posi(\labmda * T)를 따른다.
        만약, 1일 간 Posi(1/12)를 따른다면 T = 1년 일 경우, Posi(30)을 따르게 된다.
        이 때, 포아송 모수인 \lambda * T는 예상 구매 횟수를 의미
    
    2. 고객마다 일정한 기간 동안 구매하는 횟수는 다르다. 
        \lambda ~ Gamma(r, \alpha)를 따른다.

    3. j 번째 구매가 마지막이고 더 이상 구매를 하지 않을 확률, 이를 이탈률 p라고 하며,
        이탈 할 때까지의 구매 횟수는 Geo(p)를 따른다.

    4. 고객마다 더 이상 구매하지 않을 확률(이탈을)은 다르다. 이탈은 p ~ Beta(a, b)를 따른다.

    5. 고객별 일정 기간 동안의 구매 횟수와 구매를 하지 않은 확률은 서로 독립적
    
    """
    #return param_r, param_alpha, param_a, param_b
    

    #! Gamma-Gamma 모델 피팅
    # 최적의 L2 penaly "l2_gg"를 넣고, calibration data로 모형을 피팅시키고,
    # holdout data로 monetary value의 실제값과 예측값을 비교해 MSE를 계산
    spend_model = GammaGammaFitter(penalizer_coef = l2_gg)
    spend_model.fit(filtered_df['frequency_cal'], filtered_df['monetary_value_cal'])

    # conditional_expected_average_profit : 고객별 평균 구매 금액 예측
    monetary_actual = filtered_df['monetary_value_holdout']
    monetary_predicted = spend_model.conditional_expected_average_profit(filtered_df['frequency_holdout'],
                                                                         filtered_df['monetary_value_holdout']
                                                                         )
    mse = score_model(monetary_actual, monetary_predicted, 'mse')
    # print ('MSE : {0}'.format(mse))
    
    #! LTV 구하기
    final_df = whole_filtered_df.copy()
    final_df['LTV'] = spend_model.customer_lifetime_value(lifetimes_model,
                                                           final_df['frequency'],
                                                           final_df['recency'],
                                                           final_df['T'],
                                                           final_df['monetary_value'],
                                                           time = 12,
                                                           discount_rate = 0.01)
    # 전체 데이터인 whole_filtered_df에 대해서 LTV 계산
    # time, discount_rate를 향 후 몇개월 동안의 LTV를 계산할지 선택

    # BG / NBD 모형의 output인 정해진 기간만큼 예상 구매 횟수를 계산할 수 있음
    # 365일 동안 예상 구매 횟수를 구하기 위해서
    t = 365
    final_df['predicted_puchases'] = lifetimes_model.conditional_expected_number_of_purchases_up_to_time(t,
                                                                                                         final_df['frequency'],
                                                                                                         final_df['recency'],
                                                                                                         final_df['T'])
    final_df['predicted_monetary_value'] = spend_model.conditional_expected_average_profit(final_df['frequency'],
                                                                                            final_df['monetary_value'])
    final_df.to_csv(r'c:\Users\junhui\Desktop\result.csv', index=True)
    # print (final_df.sort_values(by = 'LTV'))
    
    return None



def make_entire_data(num_trading, num_customer):
    import pandas as pd
    import numpy as np
    from datetime import timedelta

    # 날짜 범위 생성
    start_date = pd.to_datetime("2022-01-01 00:00:00")
    end_date = pd.to_datetime("2022-12-31 23:59:59")
    date_range = pd.date_range(start=start_date, end=end_date, freq='S')

    # 제품 및 가격 정보
    product_ids = ['A', 'B', 'C']
    unit_prices = [30, 50, 100]

    # 고객 정보
    customer_sex = ['Female', 'Male']
    customer_age_range = range(10, 61)

    # 난수 시드 고정
    np.random.seed(42)

    # 데이터 생성
    data = {
        'InvoiceDate': np.random.choice(date_range, size=num_trading),
        'InvoiceNo': np.arange(1, num_trading + 1),
        'ProductID': np.random.choice(product_ids, size=num_trading),
        'Quantity': np.random.choice([1, 3, 6], size=num_trading, p=[0.05, 0.35, 0.6]),
        'UnitPrice': np.random.choice(unit_prices, size=num_trading),
        'CustomerSex': np.random.choice(customer_sex, size=num_trading),
        'CustomerAge': np.random.choice(customer_age_range, size=num_trading),
        'CustomerID': np.random.choice([f'Customer{i:03d}' for i in range(1, int(num_customer)+1)], size=num_trading),
    }

    # CustomerSex == 'Female'일 때, ProductID == 'B'가 가장 많도록 업데이트
    female_indices = data['CustomerSex'] == 'Female'
    data['ProductID'][female_indices] = np.random.choice(['A', 'B', 'C'], size=female_indices.sum(), p=[0.2, 0.65, 0.15])

    # CustomerAge > 20 또는 CustomerAge < 40일 때, ProductID == 'A'가 가장 많도록 업데이트
    age_indices = (data['CustomerAge'] > 20) & (data['CustomerAge'] < 40)
    data['ProductID'][age_indices] = np.random.choice(['A', 'B', 'C'], size=age_indices.sum(), p=[0.7, 0.2, 0.1])

    # 데이터 프레임 생성
    GivenDataFrame = pd.DataFrame(data)

    # 결과 확인
    print(GivenDataFrame.head())

    GivenDataFrame.to_csv(r'c:\Users\junhui\Desktop\example.csv', index=True)
    return None


def customer_data(data_size: int):
    # 정규분포를 따르는 연령 데이터 생성
    data_array = np.random.normal(30, 10, data_size)
    data_sample = pd.DataFrame(data_array)
    data_sample.columns = ["CustomerAge"]
    # '나이' 데이터를 모두 정수형으로 변경
    data_sample.CustomerAge = data_sample.CustomerAge.astype(int)
    # 10대 이하, 50대 이상 고객 통합
    data_sample.loc[data_sample["CustomerAge"] <= 10, "CustomerAge"] = 10
    data_sample.loc[data_sample["CustomerAge"] >= 50, "CustomerAge"] = 50
    # data_sample에 '성별' 변수 추가
    data_sample['CustomerSex'] = np.random.choice(['Male', 'Female'], size = data_size, p=[0.5, 0.5])
    # CustomerId
    data_sample["CustomerId"] = data_sample.index +1
    # 순서 바꾸기
    columns_order = ['CustomerId'] + [col for col in data_sample.columns if col != 'CustomerId']
    data_sample = data_sample[columns_order]
    return data_sample

def make_product_data(data_size = 1000):
    
    import pandas as pd
    import numpy as np
    from datetime import timedelta



    # 날짜 범위 생성
    start_date = pd.to_datetime("2021-01-01 00:00:00")
    end_date = pd.to_datetime("2021-12-31 23:59:59")
    date_range = pd.date_range(start=start_date, end=end_date, freq='S')

    # 제품 및 가격 정보
    product_ids = ['A', 'B', 'C']
    unit_prices = [30, 50, 100]

    # 난수 시드 고정
    np.random.seed(42)

    # 데이터 생성
    data = {
        'InvoiceDate': np.random.choice(date_range, size = int(data_size)),
        'InvoiceNo': np.arange(1, int(data_size) + 1),
        'ProductID': np.random.choice(product_ids, size = int(data_size)),
        'Quantity': np.random.choice([1, 3, 6], size = int(data_size), p=[0.05, 0.35, 0.6]),
        'UnitPrice': np.random.choice(unit_prices, size = int(data_size)),
    }
    # 데이터 프레임 생성
    GivenDataFrame = pd.DataFrame(data)

    # print (GivenDataFrame)
    return GivenDataFrame

def make_data(num_trading, num_customer):
    import pandas as pd
    import numpy as np
    from datetime import timedelta

    # 날짜 범위 생성
    start_date = pd.to_datetime("2020-01-01 00:00:00")
    end_date = pd.to_datetime("2022-11-30 23:59:59")
    date_range = pd.date_range(start=start_date, end=end_date, freq='S')

    # 제품 및 가격 정보
    product_ids = ['A', 'B', 'C']
    unit_prices = [30, 50, 100]

    # 고객 정보
    customer_sex = ['Female', 'Male']
    customer_age_range = range(10, 61)

    # 난수 시드 고정
    np.random.seed(42)

    # 생성할 고객 수만큼 고객 정보 생성
    customer_data = {
        'CustomerID': [f'Customer{i:03d}' for i in range(1, int(num_customer)+1)],
        'CustomerSex': np.random.choice(customer_sex, size=int(num_customer)),
        'CustomerAge': np.random.choice(customer_age_range, size=int(num_customer)),
    }


    # 난수 시드 초기화
    np.random.seed(42)

    # 데이터 생성
    data = {
        'InvoiceDate': np.random.choice(date_range, size=num_trading),
        'InvoiceNo': np.arange(1, num_trading + 1),
        'ProductID': np.random.choice(product_ids, size=num_trading),
        'Quantity': np.random.choice([1, 3, 6], size=num_trading, p=[0.05, 0.35, 0.6]),
        # 'UnitPrice': np.random.choice(unit_prices, size=num_trading),
        'CustomerID': np.random.choice(customer_data['CustomerID'], size=num_trading),
    }
    data_CustomerID = data['CustomerID']
    data_CustmoerSex = []
    data_CustomerAge = []
    list_CustomerID = customer_data['CustomerID']
    list_CustomerSex = customer_data['CustomerSex']
    list_CustomerAge = customer_data['CustomerAge']
    
    for row in data_CustomerID:
        data_CustmoerSex.append(list_CustomerSex[np.where(np.array(list_CustomerID) == row)[0][0]])
        data_CustomerAge.append(list_CustomerAge[np.where(np.array(list_CustomerID) == row)[0][0]])

    data['CustomerSex'] = np.array(data_CustmoerSex)
    data['CustomerAge'] = np.array(data_CustomerAge)    
    data_UnitPrice = []
    data_ProductID = data['ProductID']

    for row in data_ProductID :
        data_UnitPrice.append(unit_prices[np.where(np.array(product_ids) == row)[0][0]])
    data['UnitPrice'] = np.array(data_UnitPrice) 

    # CustomerSex == 'Female'일 때, ProductID == 'B'가 가장 많도록 업데이트
    female_indices = data['CustomerSex'] == 'Female'
    data['ProductID'][female_indices] = np.random.choice(['A', 'B', 'C'], size=female_indices.sum(), p=[0.2, 0.65, 0.15])

    # CustomerAge > 20 또는 CustomerAge < 40일 때, ProductID == 'A'가 가장 많도록 업데이트
    age_indices = (data['CustomerAge'] > 20) & (data['CustomerAge'] < 40)
    data['ProductID'][age_indices] = np.random.choice(['A', 'B', 'C'], size=age_indices.sum(), p=[0.7, 0.2, 0.1])

    # 데이터 프레임 생성
    GivenDataFrame = pd.DataFrame(data)

    # 결과 확인
    print(GivenDataFrame.head())

    GivenDataFrame.to_csv(r'c:\Users\junhui\Desktop\example.csv', index=True)
    return None

def generate_customer_data(num_customer):
    # 고객 정보
    customer_sex = ['Female', 'Male']
    customer_age_range = range(10, 61)

    # 난수 시드 고정
    np.random.seed(42)

    # 데이터 생성
    data = {
        'CustomerID': [f'Customer{i:03d}' for i in range(1, int(num_customer)+1)],
        'CustomerSex': np.random.choice(customer_sex, size=int(num_customer)),
        'CustomerAge': np.random.choice(customer_age_range, size=int(num_customer)),
    }

    # 데이터 프레임 생성
    customer_df = pd.DataFrame(data)
    
    return customer_df

if __name__ == '__main__' :
    if False : #데이터 생성하기 
        # num_trading = 10000
        # make_entire_data(num_trading)
        (num_trading, num_customer) = (500000, 4500)
        make_data(num_trading, num_customer)
    else :
        run_total_process()
    
