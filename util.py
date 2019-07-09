import os
import numpy as np
import music21
from matplotlib import pyplot as plt
import itertools
import collections
from scipy import stats
import pandas as pd


def get_file_list(filepath, file_extension, recursive=True):
    '''
    @:param filepath: a string of directory
    @:param file_extension: a string of list of strings of the file extension wanted, format in, for example, '.xml', with the ".".
    @:return A list of all directories of files in given extension in given filepath.
    If recursive is True，search the directory recursively.
    '''
    pathlist = []
    if recursive:
        for root, dirs, files in os.walk(filepath):
            for file in files:
                if type(file_extension) is list:
                    for exten in file_extension:
                        if file.endswith(exten):
                            pathlist.append(os.path.join(root, file))
                elif type(file_extension) is str:
                    if file.endswith(file_extension):
                        pathlist.append(os.path.join(root, file))
    else:
        files = os.listdir(filepath)
        for file in files:
            if type(file_extension) is list:
                for exten in file_extension:
                    if file.endswith(exten):
                        pathlist.append(os.path.join(filepath, file))
            elif type(file_extension) is str:
                if file.endswith(file_extension):
                    pathlist.append(os.path.join(filepath, file))
    if len(pathlist) == 0:
        print('Wrong or empty directory')
        raise FileNotFoundError
    return pathlist


def get_note_list(filepath):
    '''
    :param filepath: the file path of score
    :return: list of notes, in music21.Note.note
    '''
    note_list = []
    score = music21.converter.parse(filepath)
    key_word_list = ['Soprano', 'Alto', 'Tenor', 'Bass', 'S.', 'A.', 'T.', 'B.', 'Guqin']  # filter the parts
    for part in score.getElementsByClass('Part'):
        if any(key_word in part.partName for key_word in key_word_list):
            part_note_list = []
            for measure in part.getElementsByClass('Measure'):
                for element in measure:
                    if isinstance(element, music21.note.Note):
                        part_note_list.append(list(element.pitches))

                    elif isinstance(element, music21.chord.Chord):
                        part_note_list.append(list(element.pitches))

            note_list.append(part_note_list)
    return note_list


def get_interval_list(note_list):
    '''
    :param note_list: list of notes, in music21.Note.note
    :return: list of interval in int number
    '''
    interval_list = []
    for part_note_list in note_list:
        part_interval_list = []
        for i in range(len(part_note_list) - 1):
            part_interval_list.append(get_interval(part_note_list[i], part_note_list[i + 1]))
        interval_list.append(part_interval_list)
    return interval_list


def get_interval(note1, note2, fit_in_octave=True, ignore_same_pitch_chrod=False):
    if ignore_same_pitch_chrod:
        if len(set([p.midi for p in note1])) == 1:
            note1 = [note1[0]]
        if len(set([p.midi for p in note2])) == 1:
            note2 = [note2[0]]

    interval = []
    for pitch1 in note1:
        for pitch2 in note2:

            seminote = abs(pitch1.midi - pitch2.midi)
            if fit_in_octave:
                if seminote > 11 and seminote % 12 == 0:  # 大于或等于一个八度
                    seminote = 12
                else:
                    seminote = seminote % 12
            interval.append(seminote)
    return interval


def get_interval_count(interval_list):
    '''
    :param interval_list: list of intervals in int number
    :return: interval count, the count of every interval
    '''
    interval_list_merge = []
    for part in interval_list:
        interval_list_merge += list(itertools.chain.from_iterable(part))
    interval_count = dict(collections.Counter(interval_list_merge))
    return interval_count


def merge_two_dicts(x, y):
    # just add two dicts together
    return {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)}


def get_interval_probability(interval_count, bayesian_estimation=False):
    if bayesian_estimation:
        interval_count_full = np.ones((13))
    else:
        interval_count_full = np.zeros((13))
    for name, value in interval_count.items():
        interval_count_full[int(name)] += value
    interval_probability = interval_count_full / np.sum(interval_count_full)
    return interval_probability


def get_interval_transition_count(interval_list):
    interval_transition_count = np.zeros((13, 13))
    for part_interval_list in interval_list:
        for i in range(len(part_interval_list) - 1):
            interval_transition_count += get_interval_transition(part_interval_list[i], part_interval_list[i + 1])
    return interval_transition_count


def get_interval_transition(interval1, interval2):
    interval_transition = np.zeros((13, 13))
    for intv1 in interval1:
        for intv2 in interval2:
            interval_transition[intv1, intv2] += 1
    return interval_transition


def get_interval_transition_probability(interval_transition_count, bayesian_estimation=False, matrix_normalize=True):
    if bayesian_estimation:
        interval_transition_count += 1
    transfer_matrix = interval_transition_count
    if matrix_normalize:
        if np.sum(transfer_matrix) == 0:
            print('Having zero matrix')
            raise Exception
        transfer_matrix = transfer_matrix / np.sum(transfer_matrix)

    else:
        for j in range(transfer_matrix.shape[0]):
            transfer_matrix[j, :] = transfer_matrix[j, :] / sum(transfer_matrix[j, :]) if sum(
                transfer_matrix[j, :]) != 0 else 0
    return transfer_matrix


def KL_div(a, b):
    # compute D_KL(a||b) with smoothing added
    a = a.reshape(-1) + 1e-5
    b = b.reshape(-1) + 1e-5
    KL_ab = np.sum(a * np.log(a / b))
    return KL_ab


def dir_to_interval_probability(file_dir, exten='.xml'):
    '''
    directory to interval probability
    '''
    if isinstance(file_dir, list):
        file_list = []
        for fp in file_dir:
            file_list += get_file_list(fp, exten)
    elif file_dir.endswith(exten):
        file_list = [file_dir]
    else:
        file_list = get_file_list(file_dir, exten)
    interval_count = {}
    for filepath in file_list:
        note_list = get_note_list(filepath)
        interval_list = get_interval_list(note_list)
        interval_count = merge_two_dicts(interval_count, get_interval_count(interval_list))
    interval_probability = get_interval_probability(interval_count)
    return interval_probability


def show_interval_count(interval_count, title, save_fig=True, fig_size=3.6, dpi=800, filename=None, set_tile=False):
    names = [str(i) for i in range(13)]
    fig, ax = plt.subplots(figsize=(fig_size + fig_size * 0.5, fig_size), dpi=dpi)
    ax.bar(names, height=list(interval_count))
    ax.set_xlabel('Interval (semitones)')
    ax.set_ylabel('Probability (%)')
    if set_tile:
        ax.set_title(title)
    if save_fig:
        if filename == None:
            fig.savefig(title + '.png')
        else:
            fig.savefig(filename + '.png')
    plt.show()


def show_two_interval_count(interval_count1, interval_count2, title, label1=None, label2=None, save_fig=True,
                            fig_size=3.6, dpi=200, filename=None, set_tile=False):
    names = [str(i) for i in range(13)]
    index = np.arange(13)
    width = 0.4
    fig, ax = plt.subplots(figsize=(fig_size + fig_size * 0.5, fig_size * 1.05), dpi=dpi)
    ax.bar(index, height=list(interval_count1), width=width, label=label1, color='b')
    ax.bar(index + width, height=list(interval_count2), width=width, label=label2, color='g')
    ax.set_xlabel('Interval (semitones)')
    ax.set_ylabel('Probability (%)')
    ax.set_xticks(index + width / 2)
    ax.set_xticklabels(names)
    plt.legend()
    if set_tile:
        ax.set_title(title)
    if save_fig:
        if filename == None:
            fig.savefig('results/' + title + '.png')
        else:
            fig.savefig('results/' + filename + '.png')
    plt.show()


def dir_to_interval_transition_probability(file_dir, exten='.xml'):
    if isinstance(file_dir, list):
        file_list = []
        for fp in file_dir:
            file_list += get_file_list(fp, exten)
    elif file_dir.endswith(exten):
        file_list = [file_dir]
    else:
        file_list = get_file_list(file_dir, exten)
    interval_transition_count = np.zeros((13, 13))
    for filepath in file_list:
        note_list = get_note_list(filepath)
        interval_list = get_interval_list(note_list)
        interval_transition_count += get_interval_transition_count(interval_list)
    interval_transition_probability = get_interval_transition_probability(interval_transition_count)
    return interval_transition_probability


def show_heatmap(data, labels, title='', cbarlabel='', show_text=True, show_colorbar=True, cmap=None, fig_size=7.2,
                 dpi=800, save_fig=True, filename=None):
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)
    im = ax.imshow(data, cmap=cmap)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    if show_text:
        data = data.round(2)
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(j, i, data[i, j],
                               ha="center", va="center", color="w")

    if show_colorbar:
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.77)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    ax.set_title(title)
    fig.tight_layout()
    plt.show()
    if save_fig:
        if filename == None:
            fig.savefig('results/' + title + '.png')
        else:
            fig.savefig('results/' + filename + '.png')
    return fig


def cross_validation(filepath1, filepath2, process_fn):
    group_name = ['1', '2', '3', '4', '5']
    prediction_all_1 = []
    prediction_all_2 = []
    for i in range(len(group_name)):
        group = group_name[i:] + group_name[:i]
        filepath_train1 = [filepath1 + '\\' + num for num in group[1:]]
        filepath_train2 = [filepath2 + '\\' + num for num in group[1:]]
        distribution_train1 = process_fn(filepath_train1)
        distribution_train2 = process_fn(filepath_train2)
        filepath_test1 = filepath1 + '\\' + group[0]
        filepath_test2 = filepath2 + '\\' + group[0]
        prediction_1 = []
        prediction_2 = []
        for test_file in get_file_list(filepath_test1, 'xml'):
            prediction = []
            distribution_test = process_fn(test_file)
            KL_div1 = KL_div(distribution_train1, distribution_test)
            KL_div2 = KL_div(distribution_train2, distribution_test)
            prediction.append(KL_div1)
            prediction.append(KL_div2)
            prediction_1.append(prediction)
        for test_file in get_file_list(filepath_test2, 'xml'):
            prediction = []
            distribution_test = process_fn(test_file)
            KL_div1 = KL_div(distribution_train1, distribution_test)
            KL_div2 = KL_div(distribution_train2, distribution_test)
            prediction.append(KL_div1)
            prediction.append(KL_div2)
            prediction_2.append(prediction)
        prediction_all_1.append(prediction_1)
        prediction_all_2.append(prediction_2)
    return prediction_all_1, prediction_all_2


def tt_rel_manual(data1, data2):
    df = len(data1) - 1
    d = data1 - data2
    m = np.mean(d)
    s = np.std(d, ddof=1)
    t, p = stats.ttest_rel(data1, data2)
    p = p / 2
    return m, df, t, p


def get_tt_test_table(pred_1, pred_2, name1=None, name2=None):
    index = []
    for i in range(5):
        index.append(name1 + ' ' + str(i + 1))
    for i in range(5):
        index.append(name2 + ' ' + str(i + 1))

    data = []

    for i in range(5):
        m, df, t, p = tt_rel_manual(np.array(pred_1[i])[:, 0], np.array(pred_1[i])[:, 1])
        data.append([round(m, 2), df, round(t, 2), '{0:1.2e}'.format(p / 2)])
    for i in range(5):
        m, df, t, p = tt_rel_manual(np.array(pred_2[i])[:, 0], np.array(pred_2[i])[:, 1])
        data.append([round(m, 2), df, round(t, 2), '{0:1.2e}'.format(p / 2)])

    return pd.DataFrame(data, index=index, columns=['m', 'df', 't', 'p'])
