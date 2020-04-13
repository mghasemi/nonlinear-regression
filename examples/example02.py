import pandas as pd
from fbprophet import Prophet

df = pd.read_csv('../data/data.csv')
print(df.head())

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=365)
print(future.tail())

forecast = m.predict(future)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

fig1 = m.plot(forecast)
fig1.savefig('1.png')