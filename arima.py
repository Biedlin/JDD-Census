import datetime
import warnings
import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima
warnings.filterwarnings('ignore')

flow_df = pd.read_csv('flow_train.csv')
flow_df1=flow_df[flow_df['city_code']=='3f7f0ce35d6d0a08377eb2efe2189f4f']
flow_df2=flow_df[flow_df['city_code']=='c7537db4101856877ea6381d0174283c']
flow_df3=flow_df[flow_df['city_code']=='58a33c947775af5de36841c9f553317d']
flow_df4=flow_df[flow_df['city_code']=='a20d041605db832309e26c003c626719']
flow_df5=flow_df[flow_df['city_code']=='06d86ef037e4bd311b94467c3320ff38']
flow_df6=flow_df[flow_df['city_code']=='5615dc7c1af1f7dabd80bd8b8ecb1ea0']
flow_df7=flow_df[flow_df['city_code']=='ee2ff207184bf16b4a0aec0f97900c27']
#city1

flow_df1=flow_df1[flow_df1['date_dt']>=20170729]

#city2
flow_df2=flow_df2[flow_df2['date_dt']>=20170729]
#city3
flow_df3=flow_df3[flow_df3['date_dt']>=20170729]
#city4
flow_df4=flow_df4[flow_df4['date_dt']>=20170729]
#city5
flow_df5=flow_df5[flow_df5['date_dt']>=20170729]
#city6
flow_df6=flow_df6[flow_df6['date_dt']>=20170729]
#city7
flow_df7=flow_df7[flow_df7['date_dt']>=20170729]
flow_df=flow_df1.append(flow_df2).append(flow_df3).append(flow_df4).append(flow_df5).append(flow_df6).append(flow_df7)

date_dt = list()
init_date = datetime.date(2018, 3, 2)
for delta in range(15):
    _date = init_date + datetime.timedelta(days=delta)
    date_dt.append(_date.strftime('%Y%m%d'))

district_code_values = flow_df['district_code'].unique()
preds_df = pd.DataFrame()
tmp_df_columns = ['date_dt', 'city_code', 'district_code', 'dwell', 'flow_in', 'flow_out']

def auto_balance(data):
    print ("-------balance--------")
    bal_df = pd.DataFrame()
    date_dt_values = data["date_dt"].unique()
    
    for date_dt in date_dt_values:
        sub_df = data[data['date_dt'] == date_dt]
        in_sum = sub_df["flow_in"].sum()
        out_sum = sub_df["flow_out"].sum()
    
        in_rate = np.mean([in_sum,out_sum]) / in_sum
        out_rate = np.mean([in_sum,out_sum]) / out_sum
    
        sub_df["flow_in"] = sub_df["flow_in"].apply(lambda x : x * in_rate)
        sub_df["flow_out"] = sub_df["flow_out"].apply(lambda x : x * out_rate)
    
    bal_df = pd.concat([bal_df,sub_df],axis=0,ignore_index=True)
	
    return bal_df

for district_code in district_code_values:
    sub_df = flow_df[flow_df['district_code'] == district_code]
    city_code = sub_df['city_code'].iloc[0]

    predict_columns = ['dwell', 'flow_in', 'flow_out']
    tmp_df = pd.DataFrame(data=date_dt, columns=['date_dt'])
    tmp_df['city_code'] = city_code
    tmp_df['district_code'] = district_code

    for column in predict_columns:
        
		
        sub_mean = np.mean(sub_df[column])
        sub_std = np.std(sub_df[column])		
		
        sub_df[column] = sub_df[column].apply(lambda x : np.clip(x,sub_mean-1.8*sub_std,sub_mean+1.8*sub_std))
		
        ts_log = np.log(1 + sub_df[column])
        arima_model = auto_arima(ts_log, start_p=1, max_p=9, start_q=1, max_q=9, max_d=5,
                                 start_P=1, max_P=9, start_Q=1, max_Q=9, max_D=5,
                                 m=7, random_state=2018,
                                 trace=True,
                                 seasonal=True,
                                 error_action='ignore',
                                 suppress_warnings=True,
                                 stepwise=True)

        preds = arima_model.predict(n_periods=15)
        preds = pd.Series(preds)
        #对数还原
        preds = np.exp(preds) - 1
        tmp_df = pd.concat([tmp_df, preds], axis=1)

    tmp_df.columns = tmp_df_columns
    preds_df = pd.concat([preds_df, tmp_df], axis=0, ignore_index=True)

#flow_in和flow_out的平衡	
preds_df = auto_balance(preds_df)	
preds_df = preds_df.sort_values(by=['date_dt'])
preds_df.to_csv('prediction_1213.csv', index=False, header=False)