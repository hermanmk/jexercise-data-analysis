import datetime as dt
from difflib import Differ

import numpy as np
import pandas as pd


def create_relative_time_column(df):

    df['Relative_time'] = dt.timedelta(seconds=0)  # Setting all the rows to 0, but only the first row will remain 0

    for i in range(1, len(df)):
        diff = df.index[i] - df.index[i-1]
        if diff.seconds > 600:
            diff = dt.timedelta(seconds=600)
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


def patch(original, edit, start, end):
    print('Original:\n', original)
    print('Before:\n', original[:start])
    print('Inserted:\n', edit)
    print('After:\n', original[end + 1:])
    print('Start:', start)
    print('End:', end)
    print(len(original))
    print(len(edit))
    print('Test:', (start == len(original) and len(edit) == 0 and abs(end + 1) == len(original)))
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
        print(column.upper())
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
            #print('SizeMeasure:', df['SizeMeasure' + file_number].iloc[row_idx])
            print('-'*10)
            print('Index:', patched_series.index.values[row_idx])
            patched_series.iat[row_idx] = patch(patched_series.iloc[row_idx - 1], edit,
                                                int(start_series.iloc[row_idx]), int(end_series.iloc[row_idx]))
            print(patched_series.iloc[row_idx])

            # Get the number of edited characters (added, deleted, changed, moved etc.)
            character_diff_series.iat[row_idx] = get_diff_length(patched_series.iloc[row_idx - 1],
                                                                 patched_series.iloc[row_idx])

        # Naming convention for the patch column
        df['SourceCode' + file_number] = patched_series
        df['character_diff' + file_number] = character_diff_series
        print('#'*20)