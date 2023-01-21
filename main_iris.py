import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sc
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import shapiro
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.weightstats import DescrStatsW, CompareMeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ------- 1) Read data

def read_data():
    source = pd.read_csv('iris/Iris.csv')
    source = source.iloc[0:, 1:]
    source = source.rename(columns={'SepalLengthCm': 'sepal_length', 'SepalWidthCm': 'sepal_width',
                                    'PetalLengthCm': 'petal_length', 'PetalWidthCm': 'petal_width',
                                    'Species': 'species'})

    width_versicolor = source.loc[source['species'] == 'Iris-versicolor', 'sepal_width']
    width_virginica = source.loc[source['species'] == 'Iris-virginica', 'sepal_width']
    return source, width_versicolor, width_virginica


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

    print(shapiro(data_1['sepal_width']))
    print(shapiro(data_2['sepal_width']))

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
    print('Samples mean:', 'value:', samp_mean)
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
    print('Samples mean:', 'value:', samp_mean)
    print('Test: \n', 'statistic:', round(tt_1_1[0], 4), 'p-value:', round(tt_1_1[1], 4))
    print('CI: \n', 'L boundary:', round(tt_1_ci_1[0], 4), 'U boundary:', round(tt_1_ci_1[1], 4))


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

    print(shapiro(data_4['difference']))

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


# ------- 6) One-Way ANOVA

def anova(data, my_palette_1):
    virg_length = data.loc[data['species'] == 'Iris-virginica', ['sepal_length', 'species']]['sepal_length']
    versi_length = data.loc[data['species'] == 'Iris-versicolor', ['sepal_length', 'species']]['sepal_length']
    setosa_length = data.loc[data['species'] == 'Iris-setosa', ['sepal_length', 'species']]['sepal_length']

    print(shapiro(virg_length))
    print(shapiro(versi_length))
    print(shapiro(setosa_length))

    print(sc.levene(virg_length, versi_length, setosa_length, center='median'))

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
    print(sm.stats.anova_lm(model_1))
    print("\n------- TUKEY -------\n")
    print(pairwise_tukeyhsd(endog=data['sepal_length'], groups=data['species'], alpha=0.05))


# ------- 7) Levene’s test

def levene_test(df_1, df_2):
    print(sc.levene(df_1['sepal_width'], df_2['sepal_width'], center='mean'))  # skewed (not normal) distributions
    print(sc.levene(df_1['sepal_width'], df_2['sepal_width'],
                    center='median'))  # symmetric, moderate-tailed distributions
    print(sc.levene(df_1['sepal_width'], df_2['sepal_width'], center='trimmed',
                    proportiontocut=0.05))  # heavy-tailed distributions


#-------- 8) K means

def k_means():
    iris = pd.read_csv("iris/Iris.csv").iloc[:, 1:]
    x = iris.iloc[:, [0, 1, 2, 3]].values

    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1, 11), wcss)
    plt.title('Wykres osypiska')
    plt.xlabel('Liczba klastrów')
    plt.ylabel('WCSS')
    plt.show()

    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
    y_kmeans = kmeans.fit_predict(x)
    print(y_kmeans[:50])
    print(y_kmeans[50:100])
    print(y_kmeans[100:])



    plt.scatter(x[y_kmeans == 0, 1], x[y_kmeans == 0, 1], s=100, c='purple', label='Grupa 1')
    plt.scatter(x[y_kmeans == 1, 1], x[y_kmeans == 1, 1], s=100, c='orange', label='Grupa 2')
    plt.scatter(x[y_kmeans == 2, 1], x[y_kmeans == 2, 1], s=100, c='green', label='Grupa 3')


    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', label='Centroidy')
    plt.legend()

    fig = plt.figure(figsize=(15, 15))
    fig.add_subplot(111, projection='3d')
    plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=100, c='purple', label='Iris-setosa')
    plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=100, c='orange', label='Iris-versicolour')
    plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=100, c='green', label='Iris-virginica')

    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', label='Centroids')
    plt.show()




# data, s_width_versicolor, s_width_virginica = read_data()
# my_palette_1, my_palette_2 = plot_parameters()
#
# df_1, df_2 = t_test_one_plot(data)
# t_test_one_virginica(s_width_virginica)
# t_test_one_versicolor(s_width_versicolor)
#
# df_3 = t_test_two_plot(data, my_palette_2)
# t_test_two(s_width_versicolor, s_width_virginica)
#
# df_4 = t_test_two_paired_plot(data)
# t_test_two_paired(df_4)
#
# anova(data, my_palette_1)
# levene_test(df_1, df_2)

k_means()






