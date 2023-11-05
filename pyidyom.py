""" (py)IDyOM(lite)

TODO:
    - Test output differences with MUST example

Tom Kaplan: t.m.kaplan@qmul.ac.uk
"""
from enum import Enum
import copy
import itertools
import numpy as np
import operator
import pandas as pd
from collections import defaultdict, namedtuple, OrderedDict
from dataclasses import dataclass, field
from types import MethodType

from ppm import PPMC
from pyidtom_utils import flatten

def strip_nan(xs):
    """ Exclude nan records row-wise """
    if len(xs.shape) == 1:
        return xs[~np.isnan(xs)]
    else:
        return xs[~np.isnan(xs).any(axis=1)]

def is_linked(key):
    """ Check whether key is linked (tuple) or not (str) """
    return not isinstance(key, str) and isinstance(key, tuple)

def log2up(xs):
    """ Safe log2 which truncates non-zero inputs at the min float """
    if isinstance(xs, np.ndarray):
        xs[xs <= 0] = np.nextafter(0, 1)
    elif xs <= 0:
        xs = np.nextafter(0, 1)
    return np.log2(xs)


class Melody:
    """ Viewpoint generator driven from a single melody (music21.stream.Score)

    Viewpoints are "cached" as pre-generated np.ndarrays, and inverse viewpoints
    are generated dynamically based on relevant alphabets and musical context.

    :param str name: Melody identifier, e.g. file path
    :param music21.stream.Score m21score: Melody object
    :param list[music21.note.Note] notes: List of notes from flattened score
    :param dict[str] views: Cache of calculated viewpoints (target and source)
    :param dict[str] views_: Default inverse viewpoint mapping, set externally
    :param dict[str] views_stripnan: Cache of viewpoints, but without any nan values
    :param dict[str] alphabets: Alphabets per viewpoint
    """

    def __init__(self, m21score, name=''):
        """
        :param music21.stream.Score m21score: Melody object
        :param str name: Melody identifier, e.g. file path
        """
        self.name = name
        self.mel = m21score
        self.notes = list(self.mel.flat.notes)
        self.views = {}
        self.views_ = {}
        self.views_stripnan = {}
        self.alphabets = {}

    # View management
    def alphabet(self, key):
        """ Return the alphabet for a viewpoint

        :param tuple|str key: Viewpoint, e.g. 'cpitch' or ('cpitch', 'cpintref')
        :return: Alphabet of viewpoint as a set
        """
        if key not in self.alphabets:
            if is_linked(key):
                self.alphabets[key] = set(map(tuple, strip_nan(self.view(key))))
            else:
                self.alphabets[key] = set(strip_nan(self.view(key)))
        return self.alphabets[key]

    def view(self, key, nans=False):
        """ Return the alphabet for a viewpoint

        :param tuple|str key: Viewpoint, e.g. 'cpitch' or ('cpint', 'cpintref')
        :param bool nans: Exclude nans from viewpoint array
        :return: Viewpoint as np.ndarray
        """
        if key not in self.views:
            if is_linked(key):
                self.views[key] = np.column_stack([self.view(k, nans=True).tolist() for k in key])
            else:
                self.views[key] = getattr(self, key)()
            self.views_stripnan[key] = strip_nan(self.views[key])
        return self.views[key] if nans else self.views_stripnan[key]

    def set_inverse_view(self, source_view, target_view, values):
        if source_view in self.views_:
            raise NotImplementedError('Unhandled multiple source->target default inverse viewpoints!')
        self.views_[source_view + '_'] = values

    def view_(self, i, key, t_alpha, evt):
        """ Return the inverse (target) viewpoint mapping for a specific event

        :param int i: Index of the most recent musical event (evt)
        :param tuple|str key: Viewpoint, e.g. 'cpitch' or ('cpint', 'cpintref')
        :param set t_alpha: Alphabet for respective target viewpoint
        :param float evt: Most recent musical event
        :return: Inverse viewpoint as set of possible values
        """
        if is_linked(key):
            inv_views = [set(self.view_(i, k, t_alpha, evt_k)) for k, evt_k in zip(key, evt)]
            return tuple(set.intersection(*inv_views))
        else:
            inv_key = '{}_'.format(key)
            if inv_key in self.views_:
                return self.views_[inv_key][evt]
            else:
                return getattr(self, inv_key)(i, t_alpha, evt)

    #### Pitch views
    def cpitch(self):
        return np.array([n.pitch.midi for n in self.notes]).astype(float)
    def cpint(self):
        return np.insert(np.diff(self.cpitch()), 0, np.nan)
    def cpint_(self, i, cpitch_alpha, cpint):
        return (self.view('cpitch')[i-1] + cpint,)
    def contour(self):
        return np.insert(np.sign(np.diff(self.cpint())), 0, np.nan)
    def contour_(self, i, cpitch_alpha, evt):
        prev_pitch = self.view('cpitch')[i-1]
        alpha_ind = cpitch_alpha.index(prev_pitch)
        if evt == -1:
            return cpitch_alpha[:alpha_ind]
        elif evt == 0:
            return (prev_pitch,)
        else: # == 1
            return cpitch_alpha[alpha_ind+1:]

    #### Temporal views
    def ioi(self):
        onsets = [float(n.offset) for n in self.notes]
        return np.insert(np.diff(onsets), 0, 0)
    def ioiratio(self):
        return np.insert(self.ioi()[2:]/self.ioi()[1:-1], 0, [np.nan]*2)
    def ioiratio_(self, i, ioi_alpha, x):
        return (self.view('ioi')[i-1] * x,)
    def posinbar(self):
        return np.array([float(n.beat) for n in self.notes])
    def barlength(self):
        m21barlen = lambda n: n._getTimeSignatureForBeat().barDuration.quarterLength
        return np.array([float(m21barlen(n)) for n in self.notes])

class Corpus:
    """ Collection of melodies, for aggregating alphabets and their mappings

    Given a set of melodies (which each compute/cache their own views), this
    serves to pool alphabets across the entire corpus. Important for fixed
    predictive distributions, that incorporate training/test alphabets; and
    similarly for generating default inverse viewpoint mappings

    :param list[Melody] stims: Melodies for testing and evaluation
    :param dict[tuple] t_alphabets: Alphabets for each source viewpoint
    :param dict[tuple] s_alphabets: Alphabets for each source viewpont
    :param dict[int] t_alpha_lens: Lengths of target viewpoint alphabets
    :param dict[OrderedDict] t_alpha_maps: Indices of each element in target
        viewpoint alphabet within computed predictive distributions
    :param dict[dict[dict[set]]] _t_inverse: Maps from source to target
        viewpoint, e.g. self._t_inverse['barlength']['ioi'][2]
    """

    def __init__(self):
        self.stims = []
        self.t_alphabets = {}
        self.s_alphabets = {}
        self.t_alpha_lens = {}
        self.t_alpha_maps = {}
        self._t_inverse = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    def add_stimulus(self, *args, **kwargs):
        """ Add a Melody instance to this collection """
        self.stims.append(Melody(*args, **kwargs))

    def derive_alphabets(self, views):
        """ Derive source and target viewpoint alphabets from melodies

        Populates attributes ``t_alphabets``, ``s_alphabet``, ``t_alpha_lens``
        and ``t_alpha_maps``. Takes a single pass over all test and training
        melodies, converting all melodies into specified viewpoints and
        aggregating derived values. Each alphabet is converted into an ordered
        tuple, and the indices are recorded, for mapping in probability density
        functions (within PPMC).

        If inverse views aren't defined, then a default mapping function is
        created that just checks for co-occurrences of source/target values,
        e.g. given posinbar=1, return all bioi occurring at that position.

        :param dict views: Mapping between target and source viewpoints
        """
        tgt_alphabets = defaultdict(set)
        src_alphabets = defaultdict(set)

        for stim in self.stims:
            for target_view in views:
                # Aggregate alphabets
                tgt_alphabets[target_view] |= stim.alphabet(target_view)
                for source_view in views[target_view]:
                    src_alphabets[source_view] |= stim.alphabet(source_view)
                # Create two-way mapping
                for source_view in flatten(views[target_view]):
                    tview = stim.view(target_view, nans=True)
                    sview = stim.view(source_view, nans=True)
                    for i in range(tview.shape[0]):
                        t, s = tview[i], sview[i]
                        if isinstance(s, np.ndarray):
                            s = tuple(s)
                        if t != np.nan and not np.any(np.isnan(s)):
                            self._t_inverse[source_view][target_view][s].add(t)

        # Create default inverse views if fn is not defined
        for target_view in views:
            for source_view in flatten(views[target_view]):
                inv_view_fn = '{}_'.format(source_view)
                if not hasattr(Melody, inv_view_fn):
                    # TODO: Can we optimise this? Surely this is not kosher....
                    for stim in self.stims:
                        vals = dict(self._t_inverse[source_view][target_view])
                        stim.set_inverse_view(source_view, target_view, vals)

        # Convert to ordered tuples (e.g. for ordinals)
        for target_view in views:
            alpha = tuple(sorted(tgt_alphabets[target_view]))
            self.t_alphabets[target_view] = alpha
            self.t_alpha_lens[target_view] = len(alpha)
            # Mapping from alpha to numeric ordinals, and inverse, for sanity
            self.t_alpha_maps[target_view] = OrderedDict((a, i) for i, a in enumerate(alpha))
            for source_view in views[target_view]:
                self.s_alphabets[source_view] = tuple(sorted(src_alphabets[source_view]))


class ModelType(Enum):
    """ Configuration of an IDyOM model """
    STM = 'STM'
    STMplus = 'STM+'
    LTM = 'LTM'
    LTMplus = 'LTM+'
    BOTH = 'BOTH'
    BOTHplus = 'BOTH+'

@dataclass
class Model:
    """ PPM model with respective learning parameters and viewpoints

    :param ModelType mtype: Model configuration
    :param str target_view: Target viewpoint, e.g. 'cpitch' or 'ioi'
    :param str|tuple source_view: Source viewpoint, e.g. 'cpint' or ('cpint', 'cpintref')
    :param bool pretrain: Learn context counts from (pre)training stimuli
    :param bool inc: Incrementally learn context counts from test stimuli
    :param bool forget: Reset context counts from test stimuli
    :param PPMC ppm: Prediction by partial matching model
    """
    mtype: ModelType
    target_view: str
    source_view: object
    pretrain: bool
    inc: bool
    forget: bool
    ppm: PPMC

    @property
    def label(self):
        """ Create a model identifier, e.g. 'cpitch.STM.cpint-cpintref' """
        sview = '-'.join(self.source_view) if is_linked(self.source_view) else self.source_view
        return '.'.join([self.target_view, self.mtype, sview]).lower()

class IDyOMlite:
    """ Information Dynamics of Music (Pearce, 2005)-lite implementation

    Feature-light implementation of IDyOM, allowing combination of multiple-
    viewpoints in various configurations. This _basic_ implementation of IDyOM
    should be considered a starting point and not an end point for experiments;
    or an  educational example of core components of IDyOM. See package-level
    comments on what is and isn't included in pyIDyOM.


    :param set _tgt_views: Target viewpoints, e.g. {'cpitch', 'bioi'}
    :param defaultdict[defaultdict[list[Model]]] models: All targe viewpoint
        models, accessed by target viewpoint, and then model type
    """

    def __init__(self, mode, order, views, corpus, test_stims, train_stims=[], ppm_cls=PPMC):
        """
        :param ModelType mode: Type of IDyOM model (STM/STM+/LTM/LTM+/BOTH/BOTH+)
        :param int order: Bound on order of PPM models -- size of predictive context
        :param dict views: Mapping between target and source viewpoints
        :param Corpus corpus: Shared collection of Melody objects
        :param list[int] test_stims: Melody indices for evaluation, i.e. testing
        :param list[int] train_stims: Melody indices for (pre)training models
        :param PPMC ppm_cls: PPM class for underlying models, PPMC or subclass
        """
        self.mode = mode
        self._ppm_cls = ppm_cls
        self.order = order
        self.views = views
        self.corpus = corpus
        self.test_stims = test_stims
        self.train_stims = train_stims

        self._tgt_views = set(self.views.keys())
        self.models = defaultdict(lambda: defaultdict(list))

    def fit(self):
        """ Configure, (optionally) train and test this IDyOM instance """
        self._init_models()
        self._pretrain()
        return self._fit()

    def _init_models(self):
        """ Construct all necessary viewpoint models

        Populates ```models``` according to ```mode``` and ```views``` specified.
        Note that unlike IDyOM, STM pre-training is _not_ possible (for now).
        """
        def _add_model(mtype, pretrain=False, inc=False, forget=True):
            """ Add a PPMC instance for all source viewpoints """
            for target_view in views:
                for source_view in views[target_view]:
                    ppm = self._ppm_cls(self.order, self.corpus.s_alphabets[source_view])
                    model = Model(mtype, target_view, source_view, pretrain, inc, forget, ppm)
                    self.models[target_view][mtype].append(model)

        if self.mode == ModelType.STM:
            _add_model(ModelType.STM, pretrain=False, inc=True, forget=True)
        elif self.mode == ModelType.STMplus:
            _add_model(ModelType.STMplus, pretrain=False, inc=True, forget=False)
        elif self.mode == ModelType.LTM:
            _add_model(ModelType.LTM, pretrain=True, inc=False)
        elif self.mode == ModelType.LTMplus:
            _add_model(ModelType.LTMplus, pretrain=True, inc=True, forget=False)
        elif self.mode == ModelType.BOTH:
            _add_model(ModelType.STM, pretrain=False, inc=True, forget=True)
            _add_model(ModelType.LTM, pretrain=True, inc=False)
        elif self.mode == ModelType.BOTHplus:
            _add_model(ModelType.STM, pretrain=False, inc=True, forget=True)
            _add_model(ModelType.LTMplus, pretrain=True, inc=True, forget=False)
        else:
            raise ValueError('Invalid mode: {}'.format(self.mode))

    def _pretrain(self):
        """ Pretrain context counts for LTM(+) models """
        for i in self.train_stims:
            stim = self.corpus.stims[i]
            for target_view in self.models:
                for model in itertools.chain(*self.models[target_view].values()):
                    if model.pretrain:
                        model.ppm.fit(stim.view(model.source_view), learn=True)

    def _fit_model(self, stim, model, pdf_shape):
        """ Compute target PDF per event in a stimulus using a given PPMC model

        Computes PDF of source viewpoint using the given PPMC model, and then
        maps that back on to the target viewpoint.

        TODO: The inverse mapping is *very* inefficient and needs reworking!

        :param Melody stim: Stimulus to be processed
        :param PPMC model: PPM model of given source viewpoint
        :param tuple[int] pdf_shape: PDF shape for true stimulus, the target
            viewpoint might truncate it due to nan values at start (e.g. ioiratio)
        :return: Normalised PDF for stimulus as np.ndarray with the
            shape (n_events, alphabet_len)
        """
        # Source viewpoint
        sstim = stim.view(model.source_view, nans=True)
        sstim_n = strip_nan(sstim)
        # How many nan events? (for derived viewpoints)
        t_offset = sstim.shape[0]-sstim_n.shape[0]
        # PDF for source viewpoint
        pdf_m = model.ppm.fit(sstim_n, learn=model.inc, forget=model.forget)

        # Inverse mapping
        t_alpha_len = self.corpus.t_alpha_lens[model.target_view]
        t_alphabet = self.corpus.t_alphabets[model.target_view]
        t_alphabet_set = self.corpus.t_alphabets[model.target_view]
        t_alpha_map = self.corpus.t_alpha_maps[model.target_view]
        # PDF for target viewpoint
        pdf_t = np.zeros(pdf_shape)
        # Dummy value for any nan'd rows
        pdf_t[:t_offset] = 1/t_alpha_len
        # Map each event back onto target viewpoint
        for probs_i, event_i in zip(pdf_m, range(t_offset, sstim.shape[0])):
            for prob, t in zip(probs_i, self.corpus.s_alphabets[model.source_view]):
                t_vals = stim.view_(event_i, model.source_view, t_alphabet, t)
                # This might seem redundant, but adds speed where t_vals={x}|{}
                if not t_vals:
                    # Spread prob mass over all possible events
                    pdf_t[event_i] += prob/t_alpha_len
                elif len(t_vals) == 1:
                    t_val = t_vals[0]
                    if t_val in t_alphabet:
                        pdf_t[event_i][t_alpha_map[t_val]] += prob
                    else:
                        pdf_t[event_i] += prob/t_alpha_len
                else:
                    # Convert to indices in our PDF, and update
                    t_inds = [t_alpha_map[x] for x in t_vals if x in t_alphabet]
                    n_inds = len(t_inds)
                    if n_inds:
                        pdf_t[event_i][t_inds] += prob/n_inds
                    else:
                        pdf_t[event_i] += prob/t_alpha_len

        # Normalise
        pdf_t += np.nextafter(0, 1)
        return pdf_t/pdf_t.sum(axis=1).reshape(-1, 1)

    @staticmethod
    def entropy_weight(xs, b=2):
        """ Relative entropy weighting of a PDF

        :param np.ndarray xs: PDF to be weighted
        :param int b: Exponent weight on entropy ratio
        :return: Weighting for input array
        """
        H = -np.sum(xs*log2up(xs))
        Hmax = np.log2(xs.size)
        return np.power(H/Hmax, -b)

    def _geom_weight(self, pdfs):
        """ Combine PDFs using a geometric product with relative entropy weighting

        :param np.ndarray pdfs: PDFs of shape (n_pdfs, stimulus_len, alphabet_size)
        :return: Reweighted single PDF of shape (stimulus_len, alphabet_size)
        """
        ws = np.apply_along_axis(self.entropy_weight, 2, pdfs)
        ws /= ws.sum(axis=0)
        pdfs = pdfs ** ws[..., np.newaxis]
        pdfs = np.prod(pdfs, axis=0)
        pdfs *= (pdfs.sum(axis=1)**-1).reshape(-1, 1)
        return ws, pdfs

    def _fit(self):
        """ Process all test stimuli and return predictive performance

        Compute PDFs for each test melodies, building output as a pd.DataFrame
        that includes per-event PDF, IC and entropy. This includes geometric
        weighting of each viewpoint model.

        :return: pd.DataFrame containing predictions, IC and entropy per event
        """
        recs = defaultdict(list)
        dfs = {}
        for stim_i in self.test_stims:
            stim = self.corpus.stims[stim_i]
            # For each target viewpoint
            for target_view in self.views:
                # Map stimulus into its viewpoint representation
                bstim = stim.view(target_view, nans=True)
                # Results structures across model types (e.g. LTM/STM)
                pdfs = np.ones((len(self.models[target_view]), len(stim.notes), self.corpus.t_alpha_lens[target_view]))

                # For each model type (e.g. STM, LTM)
                for mt_i, (model_type, models) in enumerate(self.models[target_view].items()):
                    # Results structures across (linked) viewpoints
                    pdfs_m = np.ones((len(models), len(stim.notes), self.corpus.t_alpha_lens[target_view]))
                    # For each model (i.e. source viewpoint)
                    for m_i, model in enumerate(models):
                        pdfs_m[m_i] = self._fit_model(stim, model, pdfs.shape[1:])
                    # Combine (weighted), TODO: stash first argument, weights
                    _, pdfs_m = self._geom_weight(pdfs_m)
                    pdfs[mt_i] = pdfs_m
                # Combine (weighted), TODO: stash first argument, weights
                _, pdfs = self._geom_weight(pdfs)

                # Build output dataframe
                for event_i, (event, pdf) in enumerate(zip(bstim, pdfs)):
                    ic = -log2up(pdf[self.corpus.t_alpha_maps[target_view][event]])
                    ent = -np.sum(pdf*log2up(pdf))
                    recs[target_view].append([stim.name, event_i, event, ic, ent, *pdf])

        # Build output records
        for target_view in recs:
            cols = ['stim', 'event', 'val', 'ic', 'ent'] + list(self.corpus.t_alphabets[target_view])
            for i in range(2, len(cols)):
                cols[i] = '{}_{}'.format(target_view, cols[i])
            dfs[target_view] = pd.DataFrame(recs[target_view], columns=cols)

        # Aggregate all results
        df = pd.concat(dfs.values(), join='inner', axis=1)
        df['ic'] = df.loc[:, df.columns.str.endswith('_ic')].sum(axis=1)
        df['ent'] = df.loc[:, df.columns.str.endswith('_ent')].sum(axis=1)
        return df

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('order', type=int, help='Order of contexts')
    parser.add_argument('mels_glob', type=str, help='Melodies glob')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--csv', action='store_true')
    args = parser.parse_args()

    # Load some melodies
    from pyidyom_utils import load_corpus_music21, display, folds
    print('Loading melodies...')
    corpus = Corpus()
    for path, score in load_corpus_music21(args.mels_glob):
        corpus.add_stimulus(score, path)
    N = len(corpus.stims)

    views = {
        'cpitch': [('cpint', 'contour')],
        #'ioi': ['ioiratio'],
        'ioi': [('ioiratio', 'posinbar', 'barlength')],
    }
    corpus.derive_alphabets(views)

    print('Running cross-val...')
    dfs = []
    for i, (train, test) in enumerate(folds(N, folds=10)):
        idyom_i = IDyOMlite(ModelType.BOTHplus, args.order, views, corpus, test, train_stims=train)
        out_df = idyom_i.fit()
        dfs.append(out_df)

    idyom_df = pd.concat(dfs, ignore_index=True)
    idyom_df = idyom_df.loc[:, ~idyom_df.columns.duplicated()]

    if args.verbose:
        display(idyom_df)
    if args.csv:
        print(idyom_df.to_csv(None, index=False))

