import json
from os import stat
from os import path
from ast import literal_eval as make_tuple

import pandas as pd

all_author_names = ['abbott', 'stevenson', 'alcott', 'alger', 'allen', 'austen', 'bronte', 'cooper', 'dickens',
                    'garland', 'hawthorne', 'james', 'melville', 'page', 'thoreau', 'twain', 'doyle', 'irving', 'poe',
                    'jewett', 'wharton']


def get_active_author_names(file_name):
    if path.exists(file_name) and stat(file_name).st_size > 0:
        with open(file_name, mode='r+') as f:
            names = f.readlines()

            return [x.strip() for x in names]

    return []


def add_to_active(file_name, author_name):
    with open(file_name, mode='a+') as f:
        f.writelines([author_name + '\n'])


def delete_from_active(file_name, author_name):
    with open(file_name, mode='r+') as f:
        names = f.readlines()
    with open(file_name, mode='w+') as f:
        try:
            names = set(names)
            names.remove(author_name + '\n')
        except (ValueError, KeyError) as e:
            return
        finally:
            f.writelines(names)


# def get_author_name(active_authors_file, BASE_FILE_NAME, combinations):
#     active_author_names = get_active_author_names(active_authors_file)
#
#     for auth in all_author_names:
#         if auth in active_author_names:
#             continue
#
#         current_name = "{0}{1}.txt".format(BASE_FILE_NAME, auth)
#         if path.exists(current_name) and stat(current_name).st_size > 0:
#             with open(current_name, mode='r') as f:
#                 training_results = json.load(f)
#
#                 if len(combinations) == 0:
#                     n_features, n_shift = load_best_hyperparams(auth)
#                     combinations = [n_features, n_shift]
#
#                 if set(training_results.keys()) == set(list(map(lambda c: str(c), combinations))):
#                     continue
#
#         add_to_active(active_authors_file, auth)
#
#         return auth

def get_author_name(active_authors_file, BASE_FILE_NAME, combinations, hyperparams_path='model_comparison_df.json'):
    active_author_names = get_active_author_names(active_authors_file)

    for auth in all_author_names:
        if auth in active_author_names:
            continue

        current_name = "{0}{1}.txt".format(BASE_FILE_NAME, auth)
        if path.exists(current_name) and stat(current_name).st_size > 0:
            with open(current_name, mode='r') as f:
                training_results = json.load(f)

                if len(combinations) == 0:
                    n_features, n_shift = load_best_hyperparams(auth, hyperparams_path)
                    combinations_best = [(n_features, n_shift)]

                if set(training_results.keys()) == set(list(map(lambda c: str(c), combinations))):
                    continue
                elif set(training_results.keys()) == set(list(map(lambda c: str(c), combinations_best))):
                    if 'acc' in training_results[str(combinations_best[0])].keys() and len(list(filter(None, training_results[str(combinations_best[0])]['acc']))) >= 10:
                        continue
                    elif len(list(filter(None, training_results[str(combinations_best[0])]))) >= 10:
                        continue

        add_to_active(active_authors_file, auth)

        return auth


def load_best_hyperparams(name, path='model_comparison_df.json'):
    # load best performing hyperparameters
    comparison_df = pd.read_json(path)
    tuples = [make_tuple(x) for x in comparison_df['best_comb'].to_list()]

    author_name_comb = dict(zip(all_author_names, tuples))

    nFeatures = author_name_comb[name][0]  # F: number of output features of the only layer
    nShifts = author_name_comb[name][1]  # K: number of shift tap

    return nFeatures, nShifts
