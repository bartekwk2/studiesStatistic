import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sc
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.power import TTestPower
from statsmodels.stats.weightstats import DescrStatsW, CompareMeans


# ------- 1) Read data

def read_data():
    source = pd.read_csv('iris/Iris.csv')
    source = source.iloc[0:, 1:]
    source = source.rename(columns={'SepalLengthCm': 'sepal_length', 'SepalWidthCm': 'sepal_width',
                                    'PetalLengthCm': 'petal_length', 'PetalWidthCm': 'petal_width',
                                    'Species': 'species'})

    width_versicolor = source.loc[source['species'] == 'Iris-versicolor', 'sepal_width']
    width_virginica = source.loc[source['species'] == 'Iris-virginica', 'sepal_width']
    width_setosa = source.loc[source['species'] == 'Iris-setosa', 'sepal_width']
    return source, width_versicolor, width_virginica, width_setosa


# ------- 2) Initial parameters for plots
def plot_parameters():
    sns.set_theme(rc={'grid.linewidth': 0.6, 'grid.color': 'white',
                      'axes.linewidth': 2, 'axes.facecolor': '#ECECEC',
                      'axes.labelcolor': '#000000',
                      'xtick.color': '#000000', 'ytick.color': '#000000'})

    palette_1 = ['#00575e', '#4bafb8', 'red']
    palette_2 = ['#00575e', '#4bafb8']
    return palette_1, palette_2


# ------- 3) One Sample t-test

# ---- 3.1) Show plots
def t_test_one_plot(data):
    data_1 = data.loc[data['species'] == 'Iris-virginica', ['sepal_width', 'species']]
    data_2 = data.loc[data['species'] == 'Iris-versicolor', ['sepal_width', 'species']]

    with plt.rc_context(rc={'figure.dpi': 150, 'axes.labelsize': 9,
                            'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
                            'legend.fontsize': 8.5, 'legend.title_fontsize': 9}):
        fig_0, ax_0 = plt.subplots(2, 2, figsize=(10, 7))

        sns.kdeplot(ax=ax_0[0, 0], x=data_1['sepal_width'],
                    common_norm=True, fill=True, alpha=0.4, color='#00575e',
                    linewidth=1.5)

        ax_0[0, 0].set_title('Iris-virginica', fontsize=10, color='black')

        sns.kdeplot(ax=ax_0[0, 1], x=data_2['sepal_width'],
                    common_norm=True, fill=True, alpha=0.4, color='#4bafb8',
                    linewidth=1.5)

        ax_0[0, 1].set_title('Iris-versicolor', fontsize=10, color='black')

        ax_0[1, 0].set_visible(False)
        ax_0[1, 1].set_visible(False)

        plt.tight_layout(pad=1.5)
        plt.show()

    return data_1, data_2


# ---- 3.2) Test and confidence interval: sepal width, virginica
def t_test_one_virginica(s_width_virginica):
    tt_1_0 = sc.ttest_1samp(s_width_virginica, 3, alternative='two-sided')

    conf_lvl = 0.95
    dfs = len(s_width_virginica) - 1
    samp_mean = np.mean(s_width_virginica)
    samp_se = sc.sem(s_width_virginica)

    tt_1_ci_0 = sc.t.interval(alpha=conf_lvl, df=dfs, loc=samp_mean, scale=samp_se)

    print('Theorised mean: \n', 'value:', 3)
    print('Test: \n', 'statistic:', round(tt_1_0[0], 4), 'p-value:', round(tt_1_0[1], 4))
    print('CI: \n', 'L boundary:', round(tt_1_ci_0[0], 4), 'U boundary:', round(tt_1_ci_0[1], 4))


# ---- 3.3) Test and confidence interval: sepal width, versicolor

def t_test_one_versicolor(s_width_versicolor):
    tt_1_1 = sc.ttest_1samp(s_width_versicolor, 2.7, alternative='two-sided')

    conf_lvl = 0.95
    dfs = len(s_width_versicolor) - 1
    samp_mean = np.mean(s_width_versicolor)
    samp_se = sc.sem(s_width_versicolor)

    tt_1_ci_1 = sc.t.interval(alpha=conf_lvl, df=dfs, loc=samp_mean, scale=samp_se)

    print('Theorised mean: \n', 'value:', 2.7)
    print('Test: \n', 'statistic:', round(tt_1_1[0], 4), 'p-value:', round(tt_1_1[1], 4))
    print('CI: \n', 'L boundary:', round(tt_1_ci_1[0], 4), 'U boundary:', round(tt_1_ci_1[1], 4))


# ---- 3.4) Power analysis

def t_test_one_power():
    power_a_0 = TTestPower()
    n_obs_0 = power_a_0.solve_power(alpha=0.05, power=0.9, effect_size=0.5, nobs=None,
                                    alternative='two-sided')

    print('Number of observations: ', round(n_obs_0, 0))


# ------- 4) Two Sample t-test: Unpaired (independent) samples

# ---- 4.1) Show plots

def t_test_two_plot(data, my_palette_2):
    data_3 = data.loc[
        (data['species'] == 'Iris-virginica') | (data['species'] == 'Iris-versicolor'), ['sepal_width', 'species']]

    with plt.rc_context(rc={'figure.dpi': 150, 'axes.labelsize': 9,
                            'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
                            'legend.fontsize': 8.5, 'legend.title_fontsize': 9}):
        fig_1, ax_1 = plt.subplots(2, 2, figsize=(10, 7))  # 2 by 2 grid - making graphs look neater

        sns.kdeplot(ax=ax_1[0, 0], x=data_3['sepal_width'],
                    hue=data_3['species'], common_norm=True, linewidth=1.5,
                    fill=True, alpha=0.4, palette=my_palette_2)

        sns.stripplot(ax=ax_1[0, 1], x=data_3['species'], s=3, jitter=0.1,
                      y=data_3['sepal_width'], palette=my_palette_2, alpha=1)

        ax_1[1, 0].set_visible(False)
        ax_1[1, 1].set_visible(False)

        plt.tight_layout(pad=1.5)
        plt.show()
    return data_3


# ---- 4.2) Test and confidence interval

def t_test_two(s_width_versicolor, s_width_virginica):
    tt_2_0 = sc.ttest_ind(s_width_versicolor, s_width_virginica, equal_var=True, alternative='two-sided')

    cm = CompareMeans(DescrStatsW(s_width_versicolor), DescrStatsW(s_width_virginica))
    tt_2_ci_0 = cm.tconfint_diff(alpha=0.05, alternative='two-sided', usevar='pooled')

    print('Mean difference: \n', 'value:', round(np.mean(s_width_versicolor) - np.mean(s_width_virginica), 3))
    print('Test: \n', 'statistic:', round(tt_2_0[0], 4), 'p-value:', round(tt_2_0[1], 4))
    print('CI: \n', 'L boundary:', round(tt_2_ci_0[0], 4), 'U boundary:', round(tt_2_ci_0[1], 4))


# ---- 4.3) Power analysis

def t_test_two_power():
    power_a_1 = TTestIndPower()
    n_obs_1 = power_a_1.solve_power(alpha=0.05, power=0.9, effect_size=0.9, nobs1=None,
                                    ratio=1.0, alternative='two-sided')

    print('Number of observations: ', round(n_obs_1, 0))


# ------- 5) Two Sample t-test: Paired (dependent) samples

def t_test_two_paired_plot(data):
    data_4 = data[['sepal_width']]
    data_4 = data_4.rename(columns={'sepal_width': 'sepal_width_before'})
    data_4['sepal_width_after'] = data_4['sepal_width_before'] + (np.random.normal(size=150) + 0.2)
    data_4['difference'] = data_4['sepal_width_after'] - data_4['sepal_width_before']
    data_4.round(1).head(4)

    # ---- 5.1) Show plots
    with plt.rc_context(rc={'figure.dpi': 150, 'axes.labelsize': 9,
                            'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
                            'legend.fontsize': 8.5, 'legend.title_fontsize': 9}):
        fig_2, ax_2 = plt.subplots(2, 2, figsize=(10, 7))  # 2 by 2 grid - making graphs look neater

        sns.kdeplot(ax=ax_2[0, 0], x=data_4['difference'],
                    color='#00575e', fill=True, alpha=0.4, linewidth=1.5)

        sm.qqplot(data_4['difference'], line='q', ax=ax_2[0, 1],
                  markerfacecolor='#00575e', markeredgecolor='#00575e', markersize=3)

        ax_2[1, 0].set_visible(False)
        ax_2[1, 1].set_visible(False)

        plt.tight_layout(pad=1.5)
        plt.show()

    return data_4


# ---- 5.2) Test and confidence interval

def t_test_two_paired(df_4):
    x_before = df_4['sepal_width_before']
    x_after = df_4['sepal_width_after']

    tt_1_r_0 = sc.ttest_rel(x_before, x_after, alternative='two-sided')

    conf_lvl = 0.95
    dfs = len(x_before) - 1
    samp_mean = np.mean(x_before - x_after)
    samp_se = sc.sem(x_before - x_after)

    tt_1_r_ci_0 = sc.t.interval(alpha=conf_lvl, df=dfs, loc=samp_mean, scale=samp_se)

    print('Mean difference: \n', 'value:', round(np.mean(x_before - x_after), 4))
    print('Test: \n', 'statistic:', round(tt_1_r_0[0], 4), 'p-value:', round(tt_1_r_0[1], 4))
    print('CI: \n', 'L boundary:', round(tt_1_r_ci_0[0], 4), 'U boundary:', round(tt_1_r_ci_0[1], 4))


# ------- 6) Welch’s t-test (unequal variances)

def welch_test(s_width_versicolor, s_width_virginica):
    tt_2_1 = sc.ttest_ind(s_width_versicolor, s_width_virginica, equal_var=False, alternative='two-sided')

    cm = CompareMeans(DescrStatsW(s_width_versicolor), DescrStatsW(s_width_virginica))
    tt_2_ci_1 = cm.tconfint_diff(alpha=0.05, alternative='two-sided', usevar='unequal')

    print('Mean difference: \n', 'value:', round(np.mean(s_width_versicolor) - np.mean(s_width_virginica), 3))
    print('Test: \n', 'statistic:', round(tt_2_1[0], 4), 'p-value:', round(tt_2_1[1], 4))
    print('CI: \n', 'L boundary:', round(tt_2_ci_1[0], 4), 'U boundary:', round(tt_2_ci_1[1], 4))


# ------- 7) One-Way ANOVA

def anova(data, my_palette_1):
    data.groupby(['species'])['sepal_length'].aggregate([np.mean, np.std, pd.Series.count]).reset_index()

    with plt.rc_context(rc={'figure.dpi': 150, 'axes.labelsize': 9,
                            'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
                            'legend.fontsize': 8.5, 'legend.title_fontsize': 9}):
        fig_2, ax_2 = plt.subplots(2, 2, figsize=(10, 7))  # 2 by 2 grid - making graphs look neater

        sns.kdeplot(ax=ax_2[0, 0], x=data['sepal_length'], linewidth=1.5,
                    hue=data['species'], common_norm=True,
                    fill=True, alpha=0.4, palette=my_palette_1)

        sns.stripplot(ax=ax_2[0, 1], x=data['species'], s=3, jitter=0.1,
                      y=data['sepal_length'], palette=my_palette_1, alpha=1)

        ax_2[1, 0].set_visible(False)
        ax_2[1, 1].set_visible(False)

        plt.tight_layout(pad=1.5)
        plt.show()

    model_1 = ols('sepal_length ~ species', data=data).fit()
    print("\n------- ANOVA -------\n")
    print(sm.stats.anova_lm(model_1).round(5))
    print("\n------- TUKEY -------\n")
    print(pairwise_tukeyhsd(endog=data['sepal_length'], groups=data['species'], alpha=0.05))


# ------- 8) Comparing variances
def variance_compare(df_3, my_palette_1, my_palette_2):
    with plt.rc_context(rc={'figure.dpi': 150, 'axes.labelsize': 9,
                            'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
                            'legend.fontsize': 8.5, 'legend.title_fontsize': 9}):
        fig_4, ax_4 = plt.subplots(2, 2, figsize=(10, 7))  # 2 by 2 grid - making graphs look neater

        sns.kdeplot(ax=ax_4[0, 0], x=df_3['sepal_width'],
                    hue=df_3['species'],
                    common_norm=True,
                    fill=True, alpha=0.4, palette=my_palette_2,
                    linewidth=1.5)

        sns.pointplot(ax=ax_4[0, 1], y=df_3['sepal_width'], x=df_3['species'],
                      palette=my_palette_1,
                      estimator=np.mean, ci='sd', scale=0.5,
                      errwidth=0.5, capsize=0.15, join=False, dodge=True)

        ax_4[1, 0].set_visible(False)
        ax_4[1, 1].set_visible(False)

        plt.tight_layout(pad=1.5)
        plt.show()


# ------- 9) F-test
def f_test(df_1, df_2):
    f = np.var(df_1['sepal_width']) / np.var(df_2['sepal_width'])

    print('p-value (two-sided test):', round(2 * (1 - sc.f.cdf(f, len(df_1) - 1, len(df_2) - 1)), 3))
    print('p-value (two-sided test):', round(2 * sc.f.sf(f, len(df_1) - 1, len(df_2) - 1), 3))


# ------- 10) Levene’s test

def levene_test(df_1, df_2):
    print(sc.levene(df_1['sepal_width'], df_2['sepal_width'], center='mean'))  # skewed (not normal) distributions
    print(sc.levene(df_1['sepal_width'], df_2['sepal_width'],
                    center='median'))  # symmetric, moderate-tailed distributions
    print(sc.levene(df_1['sepal_width'], df_2['sepal_width'], center='trimmed',
                    proportiontocut=0.05))  # heavy-tailed distributions


# ------- 11) Bartlett’s test
def bartlett_test(df_1, df_2):
    print(sc.bartlett(df_1['sepal_width'], df_2['sepal_width']))


data, s_width_versicolor, s_width_virginica, s_width_setosa = read_data()
my_palette_1, my_palette_2 = plot_parameters()

df_1, df_2 = t_test_one_plot(data)
t_test_one_virginica(s_width_virginica)

t_test_one_versicolor(s_width_versicolor)
t_test_one_power()

df_3 = t_test_two_plot(data, my_palette_2)
t_test_two(s_width_versicolor, s_width_virginica)
t_test_two_power()

df_4 = t_test_two_paired_plot(data)
t_test_two_paired(df_4)

welch_test(s_width_versicolor, s_width_virginica)
anova(data, my_palette_1)
variance_compare(df_3, my_palette_1, my_palette_2)
levene_test(df_1, df_2)
bartlett_test(df_1, df_2)
