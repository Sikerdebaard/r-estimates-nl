from pathlib import Path
from tplengine import generate_md_page
from tqdm.auto import tqdm
from sklearn.utils import resample
from sklearn import metrics
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
import scipy.stats as st


random_seed_init = 0
np.random.seed(random_seed_init)


def regression_results(y_true, y_pred):
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred) 
    mse = metrics.mean_squared_error(y_true, y_pred) 
    
    if np.all(y_true > 0) and np.all(y_pred > 0):
        mean_squared_log_error = metrics.mean_squared_log_error(y_true, y_pred)
        msle = round(mean_squared_log_error,4)
    else:
        mean_squared_log_error = None
        msle = None
    median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    
    result = {
        'explained_variance': round(explained_variance,4),
        #'mean_squared_log_error': msle,
        'r2': r2,
        'MAE': mean_absolute_error,
        'MSE': mse,
        'RMSE': np.sqrt(mse),
    }
    
    return result


def compute_confidence_bootstrap(bootstrap_metric, test_metric, N_1, alpha=0.95):
    """
    Function to calculate confidence interval for bootstrapped samples.
    metric: numpy array containing the result for a metric for the different bootstrap iterations
    test_metric: the value of the metric evaluated on the true, full test set
    alpha: float ranging from 0 to 1 to calculate the alpha*100% CI, default 0.95
    """
    metric_std = np.std(bootstrap_metric)
    
    # this effectively does [test_metric - metric_std * 1.96, test_metric + metric_std * 1.96]
    CI = st.norm.interval(alpha, loc=test_metric, scale=metric_std)
    return CI


def bootstrap_regular(y_true, y_pred, bootstrap_iters=10_000, alpha=.95):
    smetrics = regression_results(y_true, y_pred)

    bootstrap_iters = 10000
    bootstrapped_metrics = {}
    for i in tqdm(list(range(0, bootstrap_iters)), desc='Bootstrap iters', leave=False):
        y_true = df_r_test['r_true'].values
        y_pred = df_r_test['r_pred'].values
        sample_ids = df_r_test.index.values
        y_true, y_pred, sample_ids = resample(y_true, y_pred, sample_ids)

        bmetrics = regression_results(y_true, y_pred)

        for k, v in bmetrics.items():
            if k not in bootstrapped_metrics:
                bootstrapped_metrics[k] = []
            bootstrapped_metrics[k].append(v)

    stats = {}
    percnt = {}
    figures = {}

    for k in bootstrapped_metrics.keys():
        bootstrap_res = compute_confidence_bootstrap(bootstrapped_metrics[k], smetrics[k], None, alpha)
        plt.hist(bootstrapped_metrics[k], bins=50)
        plt.title(k)
        figures[k] = plt.gcf()
        plt.figure()
        stats[k] = {'avg': smetrics[k], f'ci_{alpha*100}_low': bootstrap_res[0], f'ci_{alpha*100}_high': bootstrap_res[1]}
        
        percnt_low = ((100-alpha*100) / 2)
        percnt_high = (alpha*100)+percnt_low
        percnt[k] = np.percentile(bootstrapped_metrics[k], [percnt_low, percnt_high]).tolist() + [np.mean(bootstrapped_metrics[k])]
        #print('stddev', k, stats[k])
        #print('percentile', k, percnt[k])
        #print('SPMetric', k, smetrics[k])
        
    return stats
        

def download_rivm_r():
    df_rivm = pd.read_json('https://data.rivm.nl/covid-19/COVID-19_reproductiegetal.json').set_index('Date')
    df_rivm.index = pd.to_datetime(df_rivm.index)
    df_rivm.sort_index(inplace=True)
    df_rivm.index.rename('date', inplace=True)

    df_rivm = df_rivm[df_rivm.index >= '2021-01-01']

    rename = {
        'Rt_low': 'low',
        'Rt_avg': 'avg',
        'Rt_up': 'up',
    }

    vals = list(rename.values())
    return df_rivm.rename(columns=rename)[vals], [vals[0], vals[2]]

df_rivm, _ = download_rivm_r()
df_rivm

for r_est_file in tqdm(list(Path('data').glob('r_*.csv')), desc='model', leave=True):
    outdir = Path('data/model_performance') / r_est_file.stem.split('_', 1)[1]

    df_r = pd.read_csv(r_est_file, index_col=0)
    df_r.index = pd.to_datetime(df_r.index)
    df_r = df_r[df_r.index >= '2021-01-01']

    if '50%' in df_r.columns:
        # regular model
        df_r_test = df_r['50%'].rename('r_pred').to_frame().join(df_rivm['avg'].rename('r_true'), how='left').dropna()

        # we should probably use a bootstrapping method specifically for time-series data
        # but for now use regular bootstrapping on the median / mean
        bootstrapped_stats = bootstrap_regular(df_r_test['r_true'].values, df_r_test['r_pred'])
        df_stats = pd.DataFrame(bootstrapped_stats).T

    elif 'obs_ci_upper' in df_r.columns:
        # linear model
        pass
    else:
        raise ValueError('Model not recognised')
