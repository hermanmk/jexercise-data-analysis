import datetime as dt
import os
import errno
import copy
from collections import OrderedDict
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd


NS_MAP = "xmlns:map"


# Method for parsing and keeping namespaces
def parse_nsmap(file):
    """Custom parse method where we keep track of all the namespaces
    :param file:
    :return:
    """

    events = "start", "start-ns", "end-ns"

    root = None
    ns_map = []

    for event, elem in ET.iterparse(file, events):
        if event == "start-ns":
            ns_map.append(elem)
        elif event == "end-ns":
            ns_map.pop()
        elif event == "start":
            if root is None:
                root = elem
            elem.set(NS_MAP, dict(ns_map))

    return ET.ElementTree(root)


# Custom methods to get the full tag/attribute name using the namespaces
def getNS(elem, ns):
    return elem.get(NS_MAP).get(ns)


def getAttribNS(elem, ns, attr):
    return '{%s}%s' % (getNS(elem, ns), attr)


def attribWithNS(elem, ns, attr):
    return elem.attrib.get(getAttribNS(elem, ns, attr))


def tagWithNS(elem, ns, tag):
    return '{%s}%s' % (getNS(elem, ns), tag)

tag_dictionary_original = OrderedDict({
    'jdt:JdtSourceEditProposal': {
        'COUNTER': 0,
        'ATTEMPT': {
            'jdt:JdtSourceEditEvent': {
                'EVENT': 'SourceEdit',
                'ATTRIBUTES': {
                    'sizeMeasure': {
                        'COLUMN_NAME': 'SizeMeasure',
                        'CONVERT': int
                    },
                    'errorCount': {
                        'COLUMN_NAME': 'SourceEdit_ErrorCount',
                        'CONVERT': int
                    },
                    'warningCount': {
                        'COLUMN_NAME': 'WarningCount',
                        'CONVERT': int
                    }
                },
                'EDIT': {
                    'storedString': {
                        'COLUMN_NAME': 'StoredString',
                        'CONVERT': str
                    },
                    'start': {
                        'COLUMN_NAME': 'Start',
                        'CONVERT': int
                    },
                    'end': {
                        'COLUMN_NAME': 'End',
                        'CONVERT': int
                    }
                }
            }
        }
    },
    'junit:JunitTestProposal': {
        'COUNTER': 0,
        'ATTEMPT': {
            'junit:JunitTestEvent': {
                'EVENT': 'JunitTest',
                'ATTRIBUTES': {
                    'successCount': {
                        'COLUMN_NAME': 'SuccessCount',
                        'CONVERT': int
                    },
                    'failureCount': {
                        'COLUMN_NAME': 'FailureCount',
                        'CONVERT': int
                    },
                    'errorCount': {
                        'COLUMN_NAME': 'JunitTest_ErrorCount',
                        'CONVERT': int
                    },
                    'completion': {
                        'COLUMN_NAME': 'Completion',
                        'CONVERT': float
                    }
                }
            }
        }
    },
    'jdt:JdtLaunchProposal': {
        'COUNTER': 0,
        'ATTEMPT': {
            'jdt:JdtLaunchEvent': {
                'EVENT': 'Launch',
                'ATTRIBUTES': {

                }
            }
        }
    }
})


def read_df_from_xml(xml_file):
    data_list = []
    tree = parse_nsmap(xml_file)
    root = tree.getroot()
    exercise_proposals = tree.find(tagWithNS(root, 'exercise', 'ExerciseProposals'))
    if exercise_proposals is None:
        return None
    print(exercise_proposals.tag)
    tag_dictionary = copy.deepcopy(tag_dictionary_original)  # Doing a deepcopy so the counters will start from 0
    for proposalParts in exercise_proposals:
        print('Exercise part:', proposalParts.attrib.get('exercisePart'))
        for taskProposals in proposalParts:
            task_type = attribWithNS(taskProposals, 'xsi', 'type')
            print('Task type:', task_type)
            if task_type in tag_dictionary:
                task_dictionary = tag_dictionary[task_type]
                task_dictionary['COUNTER'] += 1
            else:  # F.ex: workbench:DebugEventProposal
                continue
            for attempts in taskProposals:
                print('Attempt type:', attribWithNS(attempts, 'xsi', 'type'))
                timestamp = int(attempts.attrib.get('timestamp'))
                # Dividing by 1000 to get from ms to seconds
                attempt_date_time = dt.datetime.fromtimestamp(int(timestamp / 1e3))
                print('Attempt date time:', attempt_date_time)
                data_dict = OrderedDict()
                for attempt_type_name, attempt_dict in task_dictionary['ATTEMPT'].items():
                    data_dict[attempt_dict['EVENT'] + str(task_dictionary['COUNTER'])] = None
                    for attr, attr_dict in attempt_dict['ATTRIBUTES'].items():
                            data_dict[attr_dict['COLUMN_NAME'] + str(task_dictionary['COUNTER'])] = 0
                if getAttribNS(attempts, 'xsi', 'type') not in attempts.attrib:  # Some kind of error in the xml file?
                    continue
                attempt_type = attribWithNS(attempts, 'xsi', 'type')
                if attempt_type in task_dictionary['ATTEMPT']:
                    attempt_dictionary = task_dictionary['ATTEMPT'][attempt_type]
                    data_dict[attempt_dictionary['EVENT'] + str(task_dictionary['COUNTER'])] = 1
                    for attribute, attribute_spec in attempt_dictionary['ATTRIBUTES'].items():
                        column_name = attribute_spec['COLUMN_NAME'] + str(task_dictionary['COUNTER'])
                        convert_method = attribute_spec['CONVERT']
                        if attribute in attempts.attrib:  # TODO: If attribute never occurs; no column for it!
                            data_dict[column_name] = convert_method(attempts.attrib.get(attribute))
                    # TODO: MAKE THIS NOT HARDCODED!
                    if 'EDIT' in attempt_dictionary:
                        edit = attempts.find('edit')
                        if edit is not None:
                            for edit_attrib, edit_attrib_spec in attempt_dictionary['EDIT'].items():
                                if edit_attrib in edit.attrib:
                                    column_name = edit_attrib_spec['COLUMN_NAME'] + str(task_dictionary['COUNTER'])
                                    convert_method = edit_attrib_spec['CONVERT']
                                    data_dict[column_name] = convert_method(edit.get(edit_attrib))
                data_list.append([attempt_date_time, data_dict])
            print('--------------')
        print()
    data_list.sort(key=lambda x: x[0])  # Sorting by timestamp (attempt_date_time)
    df_index = pd.DatetimeIndex([x[0] for x in data_list])
    df = pd.DataFrame([x[1] for x in data_list], index=df_index)
    return df


def save_all_xml_from_zip_to_csv(path):
    files_loaded = 0
    files_discarded = 0
    with zipfile.ZipFile(path, 'r') as myzip:
        ex_file_list = myzip.namelist()
        for exFilePath in ex_file_list:
            with myzip.open(exFilePath, 'r') as xml_file:
                files_loaded += 1
                split_path = exFilePath.split('/')
                file_name = split_path[-1][:-3]
                user_name = split_path[-2]
                exercise_name = split_path[-3]
                directory = 'data/csv/%s/%s' % (exercise_name, user_name)
                try:  # Creating the dictionary if it doesn't already exist
                    os.makedirs(directory)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
                df = read_df_from_xml(xml_file)
                if df is None or df.empty:
                    files_discarded += 1
                    print('Discarded: {}/{}'.format(user_name, file_name))
                    continue
                save_path = '%s/%s.csv' % (directory, file_name)
                df.to_csv(save_path, mode='w+')
    print('Files loaded:', files_loaded)
    print('Files discarded:', files_discarded)
    print('{:.2f}%'.format((files_loaded - files_discarded) / files_loaded * 100))


def get_df_for_xml_file(path):
    """
    Example: path='data/oving5/0141126194/Partner.ex'
    """
    with open(path, 'r') as xml_file:
        split_path = path.split('/')
        file_name = split_path[-1][:-3]
        user_name = split_path[-2]
        return read_df_from_xml(xml_file)
