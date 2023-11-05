""" Utilities for running (py)IDyOM(lite)

Tom Kaplan: t.m.kaplan@qmul.ac.uk
"""
from __future__ import annotations

from collections import Iterable

def flatten(xs):
    """ Recursively turn list (of lists) into flat list """
    if isinstance(xs, Iterable) and not isinstance(xs, str):
        return [a for i in xs for a in flatten(i)]
    else:
        return [xs]

def display(idyom_df):
    """ Tabulate the output of ```IDyOMlite.fit``` """
    import tabulate
    table = tabulate.tabulate(idyom_df, headers='keys', showindex=False,
                              tablefmt="orgtbl", floatfmt=(".3f"))
    print('\n{}\n'.format(table))

def load_corpus_music21(path_glob):
    """ Yields melodies loaded through ```music21``` from a glob """
    import glob
    import music21
    for mel_path in glob.glob(path_glob):
        yield mel_path, music21.converter.parse(mel_path)

def folds(n, folds=10, random_state=0, shuffle=True):
    """ Candidate splits of ```n``` items in ```folds``` partitions """
    from sklearn.model_selection import KFold
    return KFold(n_splits=folds, random_state=random_state, shuffle=shuffle).split(range(n))

