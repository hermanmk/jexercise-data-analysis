import datetime as dt
from difflib import Differ, SequenceMatcher

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

PAUSE_SECONDS = 600


def create_relative_time_column(df):

    df['Relative_time'] = dt.timedelta(seconds=0)  # Setting all the rows to 0, but only the first row will remain 0

    for i in range(1, len(df)):
        diff = df.index[i] - df.index[i-1]
        if diff.seconds > PAUSE_SECONDS:
            diff = dt.timedelta(seconds=PAUSE_SECONDS)
        accumulated = df.Relative_time.iloc[i-1] + diff
        df.Relative_time.iat[i] = accumulated

    #df['Relative_time_seconds'] = df.Relative_time.apply(lambda x: x.seconds)

    df['Active_time'] = df.index[0] + df.Relative_time


def fill_nan_columns(df):
    # Filling NaNs
    df.update(df.filter(regex='^SourceEdit\d{1,}$').fillna(0))  # Filling NaNs with 0 wherever there is no SourceEdit event
    df.update(df.filter(regex='^JunitTest\d{1,}$').fillna(0))  # Filling NaNs with 0 wherever there is no JunitTest event
    df.update(df.filter(regex='^Launch\d{1,}$').fillna(0))  # Filling NaNs with 0 wherever there is no Launch event

    # Forward fill
    df.update(df.filter(regex='^SizeMeasure\d{1,}$').ffill().fillna(0))  # Forward-filling all columns beginning with 'SizeMeasure'
    df.update(df.filter(regex='^WarningCount\d{1,}$').ffill().fillna(0))  # Forward-filling all columns beginning with 'WarningCount'
    df.update(df.filter(regex='^Completion\d{1,}$').ffill().fillna(0))  # Forward-filling all columns beginning with 'Completion'
    df.update(df.filter(regex='^SuccessCount\d{1,}$').ffill().fillna(0))  # Forward-filling all columns beginning with 'SuccessCount'
    df.update(df.filter(regex='^FailureCount\d{1,}$').ffill().fillna(0))  # Forward-filling all columns beginning with 'FailureCount'
    df.update(df.filter(regex='^JunitTest_ErrorCount\d{1,}$').ffill().fillna(0))  # Forward-filling all columns beginning with 'JunitTest_ErrorCount'
    df.update(df.filter(regex='^SourceEdit_ErrorCount\d{1,}$').ffill().fillna(0))  # Forward-filling all columns beginning with 'JunitTest_ErrorCount'


def aggregate_columns(df):
    # Setting SourceEdit and JunitTest to 1 where any of the SourceEdit# or JunitTest# columns are 1
    df['SourceEdit'] = 0
    df.SourceEdit.iloc[np.where(df.filter(regex='^SourceEdit\d{1,}$') == 1)[0]] = 1
    df['JunitTest'] = 0
    df.JunitTest.iloc[np.where(df.filter(regex='^JunitTest\d{1,}$') == 1)[0]] = 1
    df['Launch'] = 0
    df.Launch.iloc[np.where(df.filter(regex='^Launch\d{1,}$') == 1)[0]] = 1

    df['TotalSizeMeasure'] = df.filter(regex='^SizeMeasure\d{1,}$').sum(axis=1)
    # We get the total completion by averaging all the Completion columns. Not skipping NaNs
    df['TotalCompletion'] = df.filter(regex='^Completion\d{1,}$').mean(skipna=False, axis=1)
    # Aggregating ErrorCount columns
    df['TotalJunitTest_ErrorCount'] = df.filter(regex='^JunitTest_ErrorCount\d{1,}$').sum(axis=1)
    df['TotalSourceEdit_ErrorCount'] = df.filter(regex='^SourceEdit_ErrorCount\d{1,}$').sum(axis=1)
    # Creating a total ErrorCount column
    df['TotalErrorCount'] = df.TotalJunitTest_ErrorCount + df.TotalSourceEdit_ErrorCount
    # Create TotalRuns
    df['TotalRuns'] = df.JunitTest + df.Launch


def patch(original, edit, start, end):
    if len(edit) == 0 and start == len(original) and abs(end + 1) == len(original):
        # There is no edit, and because both start and end is the same absolute value as original str length we have no deletion
        # We just return the original to prevent duplicated code
        return original
    patched = original[:start] + edit + original[end + 1:]
    return patched


differ = Differ()


def get_diff_length(old, new):
    diff = differ.compare(old, new)
    length = 0
    for i in diff:
        if i.startswith('+') or i.startswith('-'):
            length += 1
    return length


def get_diff_length_lines(old, new):
    old, new = old.split('\n'), new.split('\n')
    junk = lambda x: x in " \t"
    s = SequenceMatcher(None, old, new)
    diff = 0
    for i in s.get_opcodes():
        tag, a1, a2, b1, b2 = i
        if 'replace' in tag:
            diff += max(len(s.a[a1:a2]), len(s.b[b1:b2]))
        if 'delete' in tag:
            diff += len(s.a[a1:a2])
        if 'insert' in tag:
            diff += len(s.b[b1:b2])
    return diff


def get_df_from_csv(path):
    """
    Example: path='data/csv/oving5/1395669706/Partner.csv'
    :param path:
    :return:
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    # Merge duplicate indices (same exact timestamps)
    return df.groupby(level=0, sort=False).first()


def patching_source_code(df):
    """
    :param df:
    :return:
    """
    for column in df.filter(regex='^StoredString\d{1,}$').columns:
        # Getting a Pandas Series object for the given column. Copy because we will alter it!
        stored_string_series = df[column].copy()
        # Removing all references to "\r" (carriage return). Hacky?
        stored_string_series = stored_string_series.str.replace('\r', '')
        # Getting integer index of first valid index row
        first_valid = stored_string_series.index.get_loc(stored_string_series.first_valid_index())
        # Creating a deepcopy Series for the patched source code
        patched_series = stored_string_series.copy()
        # Setting everything except the first value to NaN
        patched_series.iloc[first_valid + 1:] = np.nan
        # Getting the start and end columns for the current stored string column:
        # FIXME: Hacky?
        # First, check if there is more than one storedString value. If not, we don't have any 'ReplaceSubstringEdit'
        # and thus no start or end columns
        file_number = ''.join(filter(str.isdigit, column))
        if stored_string_series.count() > 1:
            start_series = df['Start' + file_number]
            end_series = df['End' + file_number]
        else:
            print('This file has only been edited once, skipping.')
            df['SourceCode' + file_number] = patched_series.ffill()
            continue
        character_diff_series = pd.Series(index=stored_string_series.index)
        line_diff_series = pd.Series(index=stored_string_series.index)
        for row_idx in range(first_valid + 1, len(patched_series)):
            edit = stored_string_series.iloc[row_idx]
            if pd.isnull(edit):
                if not pd.isnull(start_series.iloc[row_idx]) and not pd.isnull(end_series.iloc[row_idx]):
                    # We do have a start and end integer. Set edit to empty str since we are removing
                    edit = ''
                else:
                    # We just fill it with the latest patched value here to always have a value, and go to next iteration
                    patched_series.iat[row_idx] = patched_series.iloc[row_idx - 1]
                    continue
            patched_series.iat[row_idx] = patch(patched_series.iloc[row_idx - 1], edit,
                                                int(start_series.iloc[row_idx]), int(end_series.iloc[row_idx]))

            # Get the number of edited characters (added, deleted, changed, moved etc.)
            character_diff_series.iat[row_idx] = get_diff_length(patched_series.iloc[row_idx - 1],
                                                                 patched_series.iloc[row_idx])
            # Get the number of edited lines (added, deleted, changed, moved etc.)
            line_diff_series.iat[row_idx] = get_diff_length_lines(patched_series.iloc[row_idx - 1],
                                                                  patched_series.iloc[row_idx])

        # Naming convention for the patch column
        df['SourceCode' + file_number] = patched_series
        df['Character_diff' + file_number] = character_diff_series
        df['Line_diff' + file_number] = line_diff_series


def read_and_preprocess_from_csv(path):
    """This reads the DataFrame from the given CSV file and handles all the common preprocessing
    :param path:
    :return:
    """
    df = get_df_from_csv(path)
    patching_source_code(df)
    fill_nan_columns(df)
    create_relative_time_column(df)
    aggregate_columns(df)
    return df


def scale_data(df, scaler=MinMaxScaler):
    scaled_data = scaler().fit_transform(df)
    return pd.DataFrame(scaled_data, index=df.index, columns=df.columns)


def create_only_runs_df(df):
    return df[df.TotalRuns == 1]


def classify_struggling(df, minutes=5):
    """Algorithm for classifying struggling phases
    """
    max_window = dt.timedelta(seconds=minutes * 60)
    df['runs_last_5mins'] = np.nan
    df['acc_div_point'] = np.nan
    # Selects the first of either the first run or "x minutes into the assignment"
    #lookback_idx = min(df.index.get_loc(df[df.TotalRuns == 1].iloc[0].name),
    #                   df.index.get_loc(df.index[0] + max_window, method='backfill'))
    for row_idx in range(len(df)):
        cur_row = df.iloc[row_idx]
        try:
            # Find the look back index by going x minutes in the past
            lookback_idx = df.index.get_loc(cur_row.name - max_window, method='pad')
        except KeyError:
            # KeyError happens if x minutes haven't passed yet, thus we go to the next iteration
            continue
        try:
            # If there's a run since the x minute look back, use this row instead
            lookback_idx = df.index.get_loc(df.iloc[lookback_idx:row_idx][df.TotalRuns == 1].iloc[-1].name)
        except IndexError:
            # IndexError happens if there are no runs since the look back period
            pass
        # We get the last run
        lookback_row = df.iloc[lookback_idx]
        # Accumulated line diff since last run
        accumulated = df.filter(regex='^Line_diff\d{1,}$').loc[lookback_row.name:cur_row.name].sum().sum()
        # Line diff from last run
        line_diff = 0
        for sc_col in df.filter(regex='^SourceCode\d{1,}$').columns:
            line_diff += get_diff_length_lines(lookback_row[sc_col], cur_row[sc_col])
        # Number of runs in the last 5 minutes
        runs_in_period = df.loc[cur_row.name - max_window:cur_row.name].TotalRuns.sum()
        print(runs_in_period)
        if line_diff == 0:
            if pd.isnull(accumulated):
                acc_div_point = np.nan
            else:
                acc_div_point = 0
        else:
            acc_div_point = accumulated / line_diff
        print(acc_div_point)
        df['acc_div_point'].iat[row_idx] = acc_div_point
        df['runs_last_5mins'].iat[row_idx] = runs_in_period


def run_algorithm(assignment, hash_id, exercise, smoothing_window=5, save=True):
    path = 'data/csv/{}/{}/{}.csv'.format(assignment, hash_id, exercise)
    df = read_and_preprocess_from_csv(path)
    classify_struggling(df)
    scaled_df = scale_data(df[['acc_div_point', 'runs_last_5mins']].fillna(0))
    df['scaled_max'] = scaled_df.max(axis=1)
    df['scaled_max_smoothed'] = scaled_df.max(axis=1).ewm(smoothing_window).mean()
    if save:
        df.to_csv('data/algorithm_results/{}_{}.csv'.format(hash_id, exercise))
    return df


def plot_struggling(df, series_name, phase_dict=None, threshold=None, use_relative_time=True):
    series = df[series_name]
    if use_relative_time:
        series.index = df.Active_time
    plt.figure(figsize=(16, 6))
    if phase_dict is None:
        plt.plot(series, label=series_name)
    else:
        for key, slices in phase_dict.items():
            for s in slices:
                color = 'C0'
                if 'Struggling' in key:
                    color = 'C1'
                elif 'Completed' in key:
                    color = 'C2'
                plt.plot(series[s], label=key, color=color)
    if threshold is not None:
        plt.plot(series.index, [threshold]*len(series.index), label='Threshold', color='C3')
    plt.legend()
