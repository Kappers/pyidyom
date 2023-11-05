""" Prediction by partial matching implementation

Tom Kaplan: t.m.kaplan@qmul.ac.uk
"""
import copy
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
from collections.abc import Iterable

DEFAULT_MAX_ORDER = 2

def to_tuple(xs):
    """ Recursively turn list (of lists) into tuples """
    if isinstance(xs, Iterable) and not isinstance(xs, str):
        return tuple(to_tuple(x) for x in xs)
    else:
        return xs

class PPMC:
    """ PPM Method-C, using interpolated smoothing

    In order to test the output against an existing PPM implementation, see
    Peter Harrison's module: github.com/pmcharrison/ppm.

    I have tested this with a few examples, but not exhaustively, using:
        > library(ppm)
        > mod <- new_ppm_simple(alphabet_size = ?,                  <- Fill in
        >                       order_bound = ?,                    <- Fill in
        >                       exclusion = FALSE,
        >                       update_exclusion = FALSE,
        >                       escape = "c",
        >                       debug_smooth=TRUE,                  <- Detailed output
        >                       shortest_deterministic=FALSE);
    For example:
        > mod <- new_ppm_simple(alphabet_size = 5,
        >                       order_bound = 4,
        >                       ...);
        > seq_2 <- factor(c("a", "b", "r", "a", "c", "a", "d", "a", "b", "r", "a"),
        >                 levels = c("a", "b", "c", "d", "r"))
        > print(seq_2)
        > res <- model_seq(mod, seq_2)
        > print(res)
        >


    :param OrderedDict[int] alpha_map: Map from symbol to index within PDFs/counts
    :param OrderedDict[float] alpha_map_inv: Map from PDF/counts index to symbol
    :param defaultdict[np.ndarray] contexts: Counts of symbols per predictive context
    """

    RESULT_HEADERS = ('Context','Event','IC','Entropy')

    def __init__(self, order, alphabet):
        """
        :param int order: Bound on order, size of predictive context
        :param tuple alphabet: Symbol set expected and use for prediction
        """
        self.order = order
        self.alphabet = tuple(sorted(alphabet))
        self.alpha_len = len(self.alphabet)

        # Mapping from alpha to numeric ordinals, and inverse, for sanity
        self.alpha_map = OrderedDict((a, i) for i, a in enumerate(self.alphabet))
        self.alpha_map_inv = OrderedDict((i, a) for a, i in self.alpha_map.items())

        # Context -> Count of Subsequent (for alphabet)
        self.contexts = defaultdict(lambda: np.zeros(self.alpha_len))

    def informationcontent(self, pdf, e):
        return -np.log2(pdf[self.alpha_map[e]])

    def entropy(self, pdf):
        return -np.sum(pdf*np.log2(pdf))

    def pdf_for(self, ctx, normalise=True):
        """ PDF calculated for a specific predictive context

        :param tuple ctx: Predictive context, a symbol sequence
        :param bool normalise: Normalise the output distribution
        :return: PDF as np.ndarray with shape (alphabet_len, 1)
        """
        pdf = self._smoothed_pdf(ctx)
        return pdf/pdf.sum() if normalise else pdf

    def fit(self, sequence, learn=False, forget=False, verbose=False):
        """ Train on complete sequence, in ngrams bounded by self.order

        :param Iterable sequence: Stimulus symbol sequence
        :param bool learn: Update context counts from stimulus observations
        :param bool forget: Reset any short-term context counts 
        :param bool verbose: Print tabulated output, good for testing
        :return: PDFs as np.ndarray with shape (n_events, alphabet_len)
        """
        sequence = to_tuple(sequence)

        prob_dists = np.zeros((len(sequence), self.alpha_len))
        orig_contexts = self.contexts.copy()
        if verbose:
            recs = []

        for i in range(len(sequence)):
            from_ind, to_ind = max(0, i-self.order), i

            # Update counts and PDF
            ctx, evt = sequence[from_ind:to_ind], sequence[to_ind]
            pdf = self.pdf_for(ctx, normalise=True)
            prob_dists[i] = pdf

            if verbose:
                # Record surprisal for upcoming symbol (evt), given context (ctx)
                ic = self.informationcontent(pdf, evt)
                ent = self.entropy(pdf)
                recs.append([','.join(str(c) for c in ctx), evt, ic, ent])

            if learn:
                # Now, recursively update counts of context (for all orders within bound),
                # e.g. [A,B,R]->A:  []->A, [R]->A, [B,R]->A, [A,B,R]->A
                for j in reversed(range(from_ind, to_ind+1)):
                    subseq = sequence[j:i+1]
                    ctx, evt = subseq[:-1], subseq[-1]
                    self.contexts[ctx][self.alpha_map[evt]] += 1

        if verbose:
            # Pretty-print the results
            import tabulate
            table = tabulate.tabulate(recs, floatfmt=(".3f"), tablefmt="orgtbl",
                                      headers=self.RESULT_HEADERS,
                                      colalign=("right","left","center","center"))
            print('\n{}\n'.format(table))

        if learn and forget:
            # If we updated context counts online, but ```forget=True```, then reset
            self.contexts = orig_contexts

        return prob_dists

    def _smoothed_pdf(self, ctx):
        """ Interpolated smoothing to derive unnormalised PDF for context

        For a reference of PPM Method-C with interpolated smoothing, see p89 of:
            Pearce, M. T. (2005). The construction and evaluation of statistical
            models of melodic structure in music perception and composition
            (Doctoral dissertation, City University London)

        :param tuple ctx: Predictive context, a symbol sequence
        :return: Unnormalised PDF as np.ndarray with shape (alphabet_len, 1)
        """
        # Base case
        observed_alpha = np.count_nonzero(self.contexts[()])
        base_prob = 1.0 / (self.alpha_len + 1 - observed_alpha)
        pdf = np.ones(self.alpha_len) * base_prob
        
        if ctx is not None:
            # Build up suffixes
            for i in reversed(range(len(ctx)+1)):
                ctx_i = ctx[i:]
                # Find the symbols occurring after this context
                subsequent = self.contexts[ctx_i]
                subseq_set = np.count_nonzero(subsequent)
                total_subsequent = subsequent.sum()

                # Interpolated probabilities
                if total_subsequent:
                    # Weighting function
                    lam_s = total_subsequent / (total_subsequent + subseq_set)
                    # Maximum likelihood estimate
                    A = lam_s * self.contexts[ctx_i]/total_subsequent
                else:
                    # Avoid divide by zero errors
                    lam_s = 0.0
                    A = np.zeros(self.alpha_len)
                pdf = A + (1.0 - lam_s) * pdf

        return pdf

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--order', type=int, help='Order of contexts',
                        default=DEFAULT_MAX_ORDER)
    args = parser.parse_args()
    training = tuple(_ for _ in 'abracadabra')
    ppm = PPMC(args.order, alphabet=set(training))
    print('Initial learning...')
    pdfs_learning = ppm.fit(training, verbose=True, learn=True)
    print('Now, applied again...')
    pdfs_trained = ppm.fit(training, verbose=True, learn=False)

