from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score 
from pathlib import Path
from itertools import product, combinations
from tqdm.auto import tqdm
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import requests

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150 

import seaborn as sns
sns.set_style('whitegrid')


# affix random seed
import random

random.seed(0)
np.random.seed(0)


def approx_r_from_time_series(series, generation_interval, min_samples=3):
    df_iters = None

    print(f'Working on {series.name}')

    if not isinstance(generation_interval, (list, pd.Series, np.ndarray)):
        generation_interval = [generation_interval]

    dow_samples = ['W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI', 'W-SAT', 'W-SUN']
    for gen_int, dow in tqdm(tuple(product(generation_interval, dow_samples))):
        df_iter = series.rename(f'{dow}_{gen_int}').resample('D').mean().interpolate('polynomial', order=2).resample(dow, label='left', closed='left').quantile(.5).resample('D').last().interpolate('polynomial', order=2)
        df_iter = np.e**(df_iter.pct_change()*gen_int)
        
        if df_iters is None:
            df_iters = df_iter.to_frame()
        else:
            df_iters = df_iters.join(df_iter.to_frame(), how='outer')
        

    df_raw = df_iters.copy()
    df_iters = df_iters[df_iters.count(axis=1) >= min_samples * len(generation_interval)].T.describe([.05, .5, .95]).T
    #df_r = df_iters['mean'].rename('Approx_R').to_frame()

    return df_iters.round(6), df_raw.round(6)


def _test_r_series_shift(example_series, input_series):
    idx_sel = example_series.dropna().index.intersection(input_series.dropna().index)

    df_combined_r = example_series.rename('example').loc[idx_sel].to_frame().join(input_series.rename('input').loc[idx_sel])
    ret = [
        idx_sel.shape[0],
        df_combined_r.corr('pearson').at['example', 'input'],
        df_combined_r.corr('kendall').at['example', 'input'],
        df_combined_r.corr('spearman').at['example', 'input'],
        mean_squared_error(df_combined_r['input'].values, df_combined_r['example'].values),
        explained_variance_score(df_combined_r['input'].values, df_combined_r['example'].values),
        r2_score(df_combined_r['input'].values, df_combined_r['example'].values),
        mean_squared_error(df_combined_r['input'].values, df_combined_r['example'].values, squared=False),
        df_combined_r,
    ]
    return ret 

def test_metrics(true_series, predicted_series):
    ret = _test_r_series_shift(true_series, predicted_series)
    metrics = {
        'N': ret[0],
        'pearsons_r': np.round(ret[1], 6),
        'kendall': np.round(ret[2], 6),
        'spearman': np.round(ret[3], 6),
        'mse': np.round(ret[4], 6),
        'explained_variance_score': np.round(ret[5], 6),
        'r2': np.round(ret[6], 6),
        'rmse': np.round(ret[7], 6),
    }

    return metrics


def _find_best_shift_fit(example_series, input_series, shiftrange):
    shifted_corr = []
    dfs = {}
    for shift in range(*shiftrange):
        shifted_corr.append({'shift': shift, **test_metrics(example_series, input_series.shift(shift))})

    df_corr = pd.DataFrame(shifted_corr).set_index('shift')

    # find (sub-) optimal value
    #idx = df_corr['pearsons_r'].idxmax()
    
    #idx = df_corr['spearman'].idxmax()

    use = ['pearsons_r', 'kendall', 'spearman', 'r2']
    corr_metric_used = df_corr[use].max().idxmax()
    idx = df_corr[use][corr_metric_used].idxmax() 

    #ax = df_corr.plot()

    metrics = df_corr.loc[idx].to_dict() 
    metrics['shift_corr_metric'] = corr_metric_used
    #shift = int(round(corr_idx * .5 + mse_idx * .5, 0))
    shift = idx 

    return df_corr, shift, metrics, corr_metric_used, use


def shift_series_best_fit(df_base, column_base, df_shift, column_shift, shiftrange=(-21, 21)):
    #cut_len = -7*36 if df_shift.shape[0] > -7*36 else df_shift.shape[0]  # we only want metrics over recent numbers
    cut_len = 0
    df_corr, use_shift, metrics, corr_metric_used, use_corr_metrics = _find_best_shift_fit(df_base[column_base], df_shift[column_shift][cut_len:], shiftrange)

    return df_shift.shift(use_shift), use_shift, df_corr, metrics, corr_metric_used, use_corr_metrics


def calctrendline(series):
    x = np.linspace(0, series.shape[0], series.shape[0]+1)
    y = series.values

    x = sm.add_constant(x) # adding a constant

    model = sm.OLS(y, x).fit()
    predictions = model.predict(x)

    result = series.copy().rename('trendline').to_frame()
    result['trendline'] = predictions

    return result

def r_model_plotter(df_approx_r, draw_columns, fill_columns, draw_colors, fill_colors, title, subtitle, hard_ylim=None, r_is_1_color='#E63946'):
    df_plot = df_approx_r.copy()
    df_plot['R=1'] = 1

    draw_columns = ['R=1', *draw_columns]
    draw_colors = [r_is_1_color, *draw_colors]


    df_plot = df_plot[draw_columns]
    ax = df_plot.plot(color=draw_colors, figsize=(8, 6))


    color_counter = 0
    for mi, ma in fill_columns:
        ax.fill_between(df_approx_r.index, df_approx_r[mi], df_approx_r[ma], color=fill_colors[color_counter], alpha=.2)
        color_counter += 1

    ax.set_ylabel('R')
    ax.set_xlabel('Date')

    plt.suptitle(title, fontsize=16)
    ax.set_title(subtitle, fontsize=12)

    ylim_plot_max = int(round(df_plot.max().max() * 1.2, 0))

    if hard_ylim and ylim_plot_max > hard_ylim:
        ylim_plot_max = hard_ylim

    ax.set_ylim((0, ylim_plot_max))
    ax.set_yticks(np.linspace(0, ylim_plot_max, ylim_plot_max*4+1))

    return ax


def download_rivm_r():
    df_rivm = pd.read_json(download_file_with_progressbar('https://data.rivm.nl/covid-19/COVID-19_reproductiegetal.json')).set_index('Date')
    df_rivm.index = pd.to_datetime(df_rivm.index)
    df_rivm.sort_index(inplace=True)
    df_rivm.index.rename('date', inplace=True)

    df_rivm = df_rivm[df_rivm.index >= '2021-06-01']

    rename = {
        'Rt_low': 'rivm_low',
        'Rt_avg': 'R (RIVM)',
        'Rt_up': 'rivm_up',
    }

    vals = list(rename.values())
    return df_rivm.rename(columns=rename)[vals], [vals[0], vals[2]]


def download_sewage_data():
    df_sewage = pd.read_json(download_file_with_progressbar('https://data.rivm.nl/covid-19/COVID-19_rioolwaterdata.json'))

    df_sewage['Date_measurement'] = pd.to_datetime(df_sewage['Date_measurement'])
    df_sewage = df_sewage.set_index('Date_measurement')

    df_sewage.sort_index(inplace=True)
    df_sewage['RNA_flow_per_100000'] = df_sewage['RNA_flow_per_100000'].replace('', np.nan).astype(float)

    return df_sewage


def download_sewage_data_newstyle():
    df_sewage = download_sewage_data()

    print('Interpolating and combining AWZI station data')

    df_combined_interpolated = pd.DataFrame(index=pd.to_datetime([]))
    for awzi in tqdm(list(df_sewage['RWZI_AWZI_code'].unique())):
        df_combined_interpolated = df_combined_interpolated.join(df_sewage[df_sewage['RWZI_AWZI_code'] == awzi]['RNA_flow_per_100000'].sort_index().resample('D').interpolate('polynomial', order=2).rename(f'awzi_{awzi}'), how='outer')

        df_combined_interpolated = df_combined_interpolated.sort_index()

    return df_combined_interpolated.T.mean().rename('RNA_flow_per_100000').to_frame()


def download_nice_icu_data():
    df_icu = pd.read_excel(download_file_with_progressbar('https://github.com/Sikerdebaard/dutchcovid19data/raw/master/data/new-intake.xlsx'), index_col=0)
    df_icu.index = pd.to_datetime(df_icu.index)
    df_icu = df_icu.sum(axis=1).rename('icu_admissions_nice')

    return df_icu


def download_nice_hospital_data():
    df_hosp = pd.read_excel(download_file_with_progressbar('https://github.com/Sikerdebaard/dutchcovid19data/raw/master/data/hospitalized/new-intake.xlsx'), index_col=0)
    df_hosp.index = pd.to_datetime(df_icu.index)
    df_hosp = df_hosp.sum(axis=1).rename('hospital_admissions_nice')

    return df_hosp


def download_rivm_casecounts():
    df_casus = pd.read_csv(download_file_with_progressbar('https://data.rivm.nl/covid-19/COVID-19_casus_landelijk.csv'), sep=';').set_index('Date_statistics')
    df_casus.index = pd.to_datetime(df_casus.index)
    df_casus = df_casus['Province'].resample('D').count().rename('case-counts').sort_index()

    df_muni = pd.read_csv('https://data.rivm.nl/covid-19/COVID-19_aantallen_gemeente_per_dag.csv', sep=';').set_index('Date_of_publication')
    df_muni.index = pd.to_datetime(df_muni.index)
    df_muni = df_muni['Total_reported'].resample('D').sum().rename('municipality')

    return df_casus.to_frame().join(df_muni, how='outer')


def download_nursing_homes():
    df_nurs = pd.read_csv(download_file_with_progressbar('https://covid-analytics.nl/nursing-homes.csv'), index_col=0)
    df_nurs.index = pd.to_datetime(df_nurs.index)
    df_nurs = df_nurs[df_nurs.index > '2021-01-01']

    return df_nurs


def download_ggd_testing_data():
    df_ggd = pd.read_csv(download_file_with_progressbar('https://data.rivm.nl/covid-19/COVID-19_uitgevoerde_testen.csv'), sep=';')
    df_ggd['Date_of_statistics'] = pd.to_datetime(df_ggd['Date_of_statistics'])

    df_ggd = df_ggd.groupby('Date_of_statistics')['Tested_positive'].sum().sort_index().to_frame()
    df_ggd = df_ggd[df_ggd.index > '2021-01-01']

    return df_ggd


def download_file_with_progressbar(url):
    print(f'Downloading {url}')
    res = requests.get(url, stream=True)
    res.raise_for_status()
    size = int(res.headers.get('content-length', 0))
    bsize = 1024
    pbar = tqdm(total=size, unit='iB', unit_scale=True)
    retval = BytesIO() 
    for data in res.iter_content(bsize):
        pbar.update(len(data))
        retval.write(data)
    pbar.close()
    retval.seek(0)

    return retval

def gen_colors_with(color):
    return [
        #'#229922',  # trendline
        '#E4572E',  # trendline
        #'#C6B38E',  # RIVM
        color, # custom
        '#154273',  # RIVM
    ][::-1]


def combine_runs(dfs, generation_interval, min_samples=3):
    df_iters = None

    runcounter = 0
    for df in dfs:
        df = df.add_prefix(f'{runcounter}_')

        if df_iters is None:
            df_iters = df
        else:
            df_iters = df_iters.join(df, how='outer')
        runcounter += 1

    df_iters = df_iters[df_iters.count(axis=1) >= min_samples * len(generation_interval) * 2].T.describe([.05, .5, .95]).T

    return df_iters


def prep_and_plot(df_approx_r, main_col_label, df_rivm, modelname, outdir, title, subtitle, draw_colors, hard_ylim=5):
    #df_approx_r['approx_up'] = df_approx_r['mean'] + df_approx_r['std'] * 1.96
    #df_approx_r['approx_down'] = df_approx_r['mean'] - df_approx_r['std'] * 1.96
    df_approx_r['approx_up'] = df_approx_r['95%']
    df_approx_r['approx_down'] = df_approx_r['5%']
    df_approx_r[main_col_label] = df_approx_r['50%']

    df_plot = df_rivm.join(df_approx_r[[main_col_label, 'approx_up', 'approx_down']], how='outer')
    plot_weeks = 24 
    df_plot = df_plot[-(7*plot_weeks):]

    draw_cols = [x for x in df_plot.columns if '_' not in x]
    fill_cols = [
            ('rivm_low', 'rivm_up'),
            ('approx_down', 'approx_up'),
    ]

    ax = r_model_plotter(df_plot, draw_cols, fill_cols, draw_colors, draw_colors, title, subtitle, hard_ylim=hard_ylim)
    fig = ax.get_figure()
    fig.savefig(str(outdir / f'{modelname}.png'))

    
def corr_plot(df_corr, modelname, outdir):
    ax = df_corr.plot(title=f'Correlation numbers {modelname}', figsize=(8, 6), grid=True)
    fig = ax.get_figure()
    fig.savefig(str(outdir / f'corr_{modelname}.png'))


def modparams(new):
    global base_params
    return {**base_params, **new}


def calcmodel_plot_save(name, incomplete_shift, generation_interval, timeseries, df_example, example_main_col, plot_title, plot_subtitle, plot_label, output_path, draw_colors, min_samples):
    if incomplete_shift != 0:
        df_series_r, iters = approx_r_from_time_series(timeseries[:-incomplete_shift], generation_interval, min_samples=min_samples)
    else:
        df_series_r, iters = approx_r_from_time_series(timeseries, generation_interval, min_samples=min_samples)
        incomplete_shift = 0
    df_series_r, use_shift, df_corr, metrics, corr_metric_used, use_corr_metrics = shift_series_best_fit(df_example, example_main_col, df_series_r, '50%')
    metrics['shift'] = use_shift
    metrics['uses'] = name
    metrics['delay_days'] = use_shift

    df_corr.to_csv(output_path / f'corr_{name}.csv')
    corr_plot(df_corr[use_corr_metrics], name, output_path)

    prep_and_plot(df_series_r, plot_label, df_example, name, output_path, plot_title, plot_subtitle, draw_colors)

    df_series_r = df_series_r[df_series_r.index >= '2021-01-01']
    df_series_r = df_series_r[df_series_r.columns[:8]]
    df_series_r.dropna().to_csv(output_path / f'r_{name}.csv', index_label='date')

    return df_series_r, iters, metrics


#gen_int_min = 3
#gen_int_max = 7 

#gen_int_min = 3
#gen_int_max = 5 
gen_int_min = 4
gen_int_max = 4 
generation_interval = np.linspace(gen_int_min, gen_int_max, (gen_int_max-gen_int_min)*10+1)

print(f'Using generation interval {np.min(generation_interval)}-{np.max(generation_interval)}')

output_path = Path('data')

rivm_main_col = 'R (RIVM)'
df_rivm, rivm_fill_cols = download_rivm_r()

all_metrics = {}
r_iters = {}

base_params = {
    'generation_interval': generation_interval, 
    'df_example': df_rivm, 
    'example_main_col': rivm_main_col,
    'output_path': output_path,
    'plot_subtitle': '@covid_nl',
    'min_samples': 3,
}



#df_sewage = download_sewage_data()
df_sewage = download_sewage_data_newstyle()
params = {
    'name': 'sewage',
    'incomplete_shift': 0,
    'timeseries': df_sewage['RNA_flow_per_100000'].rename('sewage data'), 
    'plot_title': 'Sewage R estimate vs RIVM R',
    'plot_label': 'R (Sewage, estimate)',
    'draw_colors': gen_colors_with('#0ACC0D'),
}
df_r, iters, metrics = calcmodel_plot_save(**modparams(params))
all_metrics[params['name']] = metrics
r_iters[params['name']] = iters.shift(metrics['shift'])



df_icu = download_nice_icu_data()
params = {
    'name': 'icu',
    'incomplete_shift': 4,
    'timeseries': df_icu.rename('icu data'), 
    'plot_title': 'ICU admissions R estimate vs RIVM R',
    'plot_label': 'R (ICU admissions, estimate)',
    'draw_colors': gen_colors_with('#0ACC0D'),
}
df_r, iters, metrics = calcmodel_plot_save(**modparams(params))
all_metrics[params['name']] = metrics
r_iters[params['name']] = iters.shift(metrics['shift'])


df_hospital = download_nice_hospital_data()
params = {
    'name': 'hospital',
    'incomplete_shift': 4,
    'timeseries': df_hospital.rename('hospital data'), 
    'plot_title': 'Hospital admissions R estimate vs RIVM R',
    'plot_label': 'R (hospital admissions, estimate)',
    'draw_colors': gen_colors_with('#0ACC0D'),
}
df_r, iters, metrics = calcmodel_plot_save(**modparams(params))
all_metrics[params['name']] = metrics
r_iters[params['name']] = iters.shift(metrics['shift'])

df_case = download_rivm_casecounts()
params = {
    'name': 'case-counts',
    'incomplete_shift': 7,
    'timeseries': df_case['case-counts'].rename('case-counts data'), 
    'plot_title': 'Casecounts R estimate vs RIVM R',
    'plot_label': 'R (casecounts, estimate)',
    'draw_colors': gen_colors_with('#0ACC0D'),
}
df_r, iters, metrics = calcmodel_plot_save(**modparams(params))
all_metrics[params['name']] = metrics
r_iters[params['name']] = iters.shift(metrics['shift'])

params = {
    'name': 'municipal-case-counts',
    'incomplete_shift': 0,
    'timeseries': df_case['municipality'].rename('municipal case-counts data'), 
    'plot_title': 'Municipal casecounts R estimate vs RIVM R',
    'plot_label': 'R (municipal casecounts, estimate)',
    'draw_colors': gen_colors_with('#0ACC0D'),
}
df_r, iters, metrics = calcmodel_plot_save(**modparams(params))
all_metrics[params['name']] = metrics
r_iters[params['name']] = iters.shift(metrics['shift'])

df_ggd = download_ggd_testing_data()
params = {
    'name': 'ggd-positive-tests',
    'incomplete_shift': 0,
    'timeseries': df_ggd['Tested_positive'].rename('GGD positive tests'), 
    'plot_title': 'GGD positive tests R estimate vs RIVM R',
    'plot_label': 'R (GGD pos. tests, estimate)',
    'draw_colors': gen_colors_with('#0ACC0D'),
}
df_r, iters, metrics = calcmodel_plot_save(**modparams(params))
all_metrics[params['name']] = metrics
r_iters[params['name']] = iters.shift(metrics['shift'])


df_nurs = download_nursing_homes()
params = {
    'name': 'nursing-homes',
    'incomplete_shift': 7,
    'timeseries': df_nurs['Total_cases_reported'].rename('nursing-homes data'), 
    'plot_title': 'Nursing homes casecounts R estimate vs RIVM R',
    'plot_label': 'R (nursing homes casecounts, estimate)',
    'draw_colors': gen_colors_with('#0ACC0D'),
}
df_r, iters, metrics = calcmodel_plot_save(**modparams(params))
all_metrics[params['name']] = metrics
r_iters[params['name']] = iters.shift(metrics['shift'])


combo_models = {}
r_combo_models = {}
try_models = []
for num_models in range(2, len(r_iters.keys()) + 1):
    for include_models in combinations(r_iters.keys(), num_models):
        iters = [r_iters[m].copy() for m in include_models]
        try_models.append([iters, *include_models])


for iters in tqdm(try_models):
    df_combined_r = combine_runs(iters[0], generation_interval)
    metrics = test_metrics(df_rivm[rivm_main_col], df_combined_r['50%'])
    k = ', '.join(tuple(sorted(iters[1:])))
    combo_models[k] = metrics
    r_combo_models[k] = df_combined_r

df_combo_metrics = pd.DataFrame(combo_models).T.to_csv(output_path / 'combo_metrics.csv')

# we could do some model selection based on the metrics above, but for now use all

k = ', '.join(tuple(sorted(r_iters.keys())))
df_combined_r = r_combo_models[k]
metrics = combo_models[k]
metrics['shift'] = 0
metrics['uses'] = k
all_metrics['combined'] = metrics
plot_title = 'Combined R estimate vs RIVM R' 
subtitle = base_params['plot_subtitle']
prep_and_plot(df_combined_r, 'R (combined, estimate)', df_rivm, 'combined', output_path, plot_title, subtitle, gen_colors_with('#0ACC0D'))
df_combined_r.dropna().to_csv(output_path / 'r_combined.csv', index_label='date')

df_metrics = pd.DataFrame(all_metrics).T
df_metrics.round(3).to_csv(output_path / 'metrics.csv')
