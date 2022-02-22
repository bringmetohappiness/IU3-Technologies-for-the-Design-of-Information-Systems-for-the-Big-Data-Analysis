"""Здесь содержатся разнообразные утилиты для визуализации."""
import matplotlib.pyplot as plt
import numpy as np


def plot_pies(dataframe, feature, label_column, *, nrows=None, ncols=None):
    """Визуализирует распределение значений признака в виде круговой диаграммы.

    Args:
        dataframe: DataFrame, из которого будет вытаскиваться распределение.
        feature: признак, для которого визуализируется распределение ответов.
        label_column: название столбца с метками.
        nrows: количество строчек на рисунке.
        ncols: количество столбцов на рисунке.
    """
    df = dataframe.copy()
    if df[feature].isnull().sum():
        df.replace({feature: {np.NaN: '-'}}, inplace=True)

    labels = sorted(list(set(df[label_column].tolist())))
    if nrows is None or ncols is None:
        nrows = 1
        ncols = len(labels) + 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 5))
    axes = axes.flat
    fig.suptitle(feature, fontsize=16)
    # отдельные пироги по классам
    for label, ax in zip(labels, axes[:len(labels)]):
        ax.set_title(label)
        values = set(df[feature].tolist())
        sizes = [
            df[(df[label_column] == label) & (df[feature] == value)].shape[0]
            for value in values
        ]
        ax.pie(sizes, autopct='%1.1f%%')
    # пирог для всего датасета
    axes[-1].set_title('целый датасет')
    values = set(df[feature].tolist())
    sizes = [df[df[feature] == value].shape[0] for value in values]
    wedges, _, _ = axes[-1].pie(sizes, autopct='%1.1f%%')
    axes[-1].legend(wedges, values, bbox_to_anchor=(1, 0, 0.5, 1))

    plt.show()


def plot_hists(df, feature, label_column, *, bins=None, xlim, nrows=None, ncols=None):
    """Визуализирует гистограммы значений признака.

    Args:
        df: DataFrame, из которого будет вытаскиваться гистограммы.
        feature: вопрос, для которого визуализируются гистограммы.
        label_column: название столбца с метками.
        bins: количество бинов гистограмм.
        xlim (tuple): пределы шкалы по x.
        nrows: количество строчек на рисунке.
        ncols: количество столбцов на рисунке.
    """
    labels = sorted(list(set(df[label_column].tolist())))
    if nrows is None or ncols is None:
        nrows = 1
        ncols = len(labels) + 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 5), sharey=True)
    axes = axes.flat
    fig.suptitle(feature, fontsize=16)
    # отдельные гистограммы по классам
    for label, ax in zip(labels, axes[:len(labels)]):
        ax.hist(df.loc[df[label_column] == label, feature].tolist(), bins=bins)
        ax.set_title(label)
        ax.set_xlim(*xlim)
    # гистограмма для всего датасета
    axes[-1].hist(df[feature].tolist(), bins=bins)
    axes[-1].set_title('целый датасет')
    axes[-1].set_xlim(*xlim)

    plt.show()
