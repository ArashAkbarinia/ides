import numpy as np

from scipy import stats


def find_paired_rows(df, columns):
    paired_rows = []
    for row_i_ind, row_i_val in df.iterrows():
        for row_j_ind, row_j_val in df.iterrows():
            if row_i_ind < row_j_ind:
                if np.all(row_i_val[columns] == row_j_val[columns]):
                    paired_rows.append(row_j_ind, row_i_ind)
    return paired_rows


def sort_paired_rows(df, paired_rows, column, value):
    selected_pairs = []
    for p_ind, p_row in enumerate(paired_rows):
        isval0 = df.loc[p_row[0], column] == value
        isval1 = df.loc[p_row[1], column] == value
        if isval0:
            selected_pairs.append(p_row)
        elif isval1:
            selected_pairs.append(p_row[::-1])
    paired_rows = np.array(selected_pairs)
    return paired_rows


def find_matched_rows(ref_row, df, exclude=None):
    matched_rows = []
    for row_ind, row_val in df.iterrows():
        if exclude is not None and row_ind <= exclude:
            continue
        if np.all(ref_row == df.iloc[row_ind, :]):
            matched_rows.append(row_ind)
    return matched_rows


def report_effect(gain, effect_name):
    s, p = stats.wilcoxon(list(gain), alternative='greater')
    significance = ""
    for p_ths in [0.05, 0.01, 0.001]:
        if p <= p_ths:
            significance += '*'
    print(f"{effect_name} gain: {np.mean(gain):.3f} – {p:.0e} {significance} [N={len(gain)}]")
    return p


def filter_df(df, column, filter_vals):
    if type(filter_vals) == str:
        filter_vals = [filter_vals]
    new_df = df[df[column].isin(filter_vals)]
    new_df = new_df.reset_index(drop=True)
    return new_df


def compute_language_effect(df, filter_arch=None, identical_arch=True):
    df_filtered = filter_df(df, 'pre-layer', ['visual', 'visual-language'])

    identical_columns = ['pre-arch', 'pre-db', 'eeg-sampling', 'instance']

    if filter_arch is not None:
        df_filtered = filter_df(df_filtered, 'eeg-arch', filter_arch)
    elif identical_arch:
        identical_columns.append('eeg-arch')

    paired_rows = find_paired_rows(df_filtered, identical_columns)
    paired_rows = np.array(paired_rows)

    df_visuallanguage = df_filtered.iloc[paired_rows[:, 0]]  # visual-language
    df_visual = df_filtered.iloc[paired_rows[:, 1]]  # visual

    gain = df_visuallanguage['avg'].to_numpy() - df_visual['avg'].to_numpy()
    report_effect(gain, 'Multimodal visual-language')


def compute_resampling_effect(df, filter_arch=None, identical_arch=True):
    identical_columns = ['pre-arch', 'pre-db', 'pre-layer', 'instance']

    df_filtered = df
    if filter_arch is not None:
        df_filtered = filter_df(df_filtered, 'eeg-arch', filter_arch)
    elif identical_arch:
        identical_columns.append('eeg-arch')

    paired_rows = find_paired_rows(df_filtered, identical_columns)
    paired_rows = np.array(paired_rows)
    print(paired_rows.shape)

    df_baseline = df_filtered.iloc[paired_rows[:, 0]]
    df_resampling = df_filtered.iloc[paired_rows[:, 1]]

    gain = df_resampling['avg'].to_numpy() - df_baseline['avg'].to_numpy()
    report_effect(gain, 'Resampling expansion space')
