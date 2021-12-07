import pandas as pd
import statsmodels.api as sm
import numpy as np

from pathlib import Path


outdir = Path('data')


def download_rivm_r():
    df_rivm = pd.read_json('https://data.rivm.nl/covid-19/COVID-19_reproductiegetal.json').set_index('Date')
    df_rivm.index = pd.to_datetime(df_rivm.index)
    df_rivm.sort_index(inplace=True)
    df_rivm.index.rename('date', inplace=True)

    df_rivm = df_rivm[df_rivm.index >= '2021-06-01']

    rename = {
        'Rt_low': 'rivm_low',
        'Rt_avg': 'rivm_mean',
        'Rt_up': 'rivm_up',
    }

    df_rivm = df_rivm[df_rivm.index > '2021-01-01']

    vals = list(rename.values())
    return df_rivm.rename(columns=rename)[vals]


dfs = {}
df_combined = None
for csv in Path('data').glob('r_*.csv'):
    if any([x in csv.name for x in ('linear',)]):
        continue

    print(csv)
    df = pd.read_csv(csv, index_col=0)
    df.index = pd.to_datetime(df.index)
    dfs[csv.stem] = df

    if df_combined is None:
        df_combined = df['50%'].rename(csv.stem).to_frame()
    else:
        df_combined = df_combined.join(df['50%'].rename(csv.stem), how='outer')

df_rivm = download_rivm_r()

sel = df_rivm.dropna().index.intersection(df_combined.dropna().index)

x = df_combined.loc[sel]
y = df_rivm['rivm_mean'].loc[sel]

x = sm.add_constant(x) # adding a constant

model = sm.OLS(y, x).fit()
predictions = model.predict(x)

summary = model.summary()
print(summary)
print(pd.DataFrame(model.params, columns=['coef']))

#print(model.bse)
print(model.scale**.5)
print(np.sqrt(model.mse_resid))

results_as_html = summary.tables[1].as_html()
df_m_res = pd.read_html(results_as_html, header=0, index_col=0)[0]
print(df_m_res)
df_m_res.to_csv(outdir / 'linear_model_summary.csv')


x = df_combined.dropna()
x = sm.add_constant(x) # adding a constant
pred = model.get_prediction(x)

df_summary = pred.summary_frame(alpha=0.05)

print(df_summary)

df_summary.to_csv(outdir / 'r_linear_model.csv')
