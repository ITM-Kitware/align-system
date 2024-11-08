from sklearn.neighbors import KernelDensity
import numpy as np
import pickle
import codecs
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from scipy.integrate import trapezoid

from swagger_client.models import KDMAValue

KDE_MAX_VALUE = 1.0 # Value ranges from 0 to 1.0
KDE_BANDWIDTH = 0.75 * (KDE_MAX_VALUE / 10.0)


def load_kde(target_kdma, norm='globalnorm', priornorm_factor=0.5):
    if isinstance(target_kdma, KDMAValue):
        target_kdma = target_kdma.to_dict()

    if norm == 'globalnorm':
        target_kde = kde_from_base64(target_kdma['kdes']['globalnorm']['kde'])
    elif norm == 'localnorm':
        target_kde = kde_from_base64(target_kdma['kdes']['localnorm']['kde'])
    elif norm == 'rawscores':
        target_kde = kde_from_base64(target_kdma['kdes']['rawscores']['kde'])
    elif norm == 'priornorm':
        # Load target KDE and get density
        kde = kde_from_base64(target_kdma['kdes']['rawscores']['kde'])
        linspace = np.linspace(0, 1, 1000)
        density = _kde_to_pdf(kde, linspace)

        # Get prior KDE and density
        #TODO - get prior data counts from train yamls rather than precomputing and hard coding
        if target_kdma['kdma'] == 'QualityOfLife':
            prior_data = [0.1]*24 + [0.3]*12 + [0.7]*12 + [0.9]*24
        elif target_kdma['kdma'] == 'PerceivedQuantityOfLivesSaved':
            prior_data = [0.1]*24 + [0.3]*16 + [0.7]*8 + [0.9]*24
        else:
            raise RuntimeError(f'No prior data for {target_kde["kdma"]}')
        prior_kde = get_kde_from_samples(prior_data)
        prior_density = _kde_to_pdf(prior_kde, linspace)

        # Normalize the target density based on prior density
        normalized_density = density / (prior_density + 1e-10)
        # Weight the normalization - linear combo of original density and normalized density
        normalized_density = (priornorm_factor*normalized_density) + ((1-priornorm_factor)*density)

        # Use normalized density to construct normalized KDE from samples
        norm_samples = []
        for value in [0.1, 0.3, 0.7, 0.9]:
            frequency = normalized_density[int(value*len(linspace))]
            norm_samples += [value]*round(frequency*100) # Count = freq *100 for smoothness
        target_kde = get_kde_from_samples(norm_samples)
    else:
        raise RuntimeError(norm, "normalization distribution matching not implemented.")
    return target_kde


##### Reference: https://github.com/ITM-Soartech/ta1-server-mvp/blob/dre/submodules/itm/src/itm/kde.py

def sample_kde():
    """
    Generates a random KDMA Measurement based on a
    normally distributed random sample

    The normal distribution is centered on `norm_loc` with a
    a scale of `norm_scale`
    """
    #X = np.array(X) # convert to numpy (if not already)
    N = 100
    X = np.random.normal(0, 1, int(0.3 * N))

    kde = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(X[:, np.newaxis])

    return kde

def kde_to_base64(kde: KernelDensity) -> str:
    return codecs.encode(pickle.dumps(kde), "base64").decode()

def kde_from_base64(base64_str: str) -> KernelDensity:
    return pickle.loads(codecs.decode(base64_str.encode(), "base64"))

#### Based on: https://github.com/ITM-Soartech/ta1-server-mvp/blob/009afe4b3548c598f83994eba2611709b8c10a0a/submodules/itm/src/itm/kdma_profile.py#L69

def get_kde_from_samples(X: list[float]):
    """
    Generates a KDE based on a sample X
    """
    X = np.array(X) # convert to numpy (if not already)
    kde = KernelDensity(kernel="gaussian", bandwidth=KDE_BANDWIDTH).fit(X[:, np.newaxis])
    return kde

######### Ref: https://github.com/ITM-Soartech/ta1-server-mvp/blob/009afe4b3548c598f83994eba2611709b8c10a0a/submodules/itm/src/itm/alignment/similarity_functions.py
def _normalize(x, y):
    """
    Normalize probability distribution y such that its integral over domain x is 1.

    Parameters
    ----------
    x: ndarray
        domain over which discrete probability distribution y is defined.

    y: ndarray
        probability distribution at each point in x. Y is proportional to the
        probability density of the distribution at x.

    Returns
    --------
    pdf: ndarray
        array with same shape as y that gives normalized probability density function
        values at each point x.

    """
    # area under curve
    auc = trapezoid(y, x)

    # scale y by auc so that new area under curve is 1 --> probability density
    pdf = y / auc

    return pdf


def _kde_to_pdf(kde, x, normalize=True):
    """
    Evaluate kde over domain x and optionally normalize results into pdf.

    Parameters
    ----------
    kde: sklearn KDE model
        model used to generate distribution.

    x: ndarray
        points to evaulate kde at to generate probability function.


    Returns
    ---------
    pf: ndarray
        array containing probability function evaluated at each element in x.

    """
    pf = np.exp(kde.score_samples(x[:,np.newaxis]))

    if normalize:
        pf = _normalize(x, pf)

    return pf


def hellinger_similarity(kde1, kde2, samples: int):
    """
    Similarity score derived from the Hellinger distance.

    The Hellinger similarity :math:`H(P,Q)`  between probability density functions
    :math:`P(x)` and :math:`Q(x)` is given by:

    .. math::
        H(P,Q) = 1 - D(P,Q)

    Where :math:`D(P,Q)` is the
    `hellinger distance <https://en.wikipedia.org/wiki/Hellinger_distance>`_ between
    the distributions.

    The similarity score is bounded between 0 (:math:`P` is 0 everywhere where
    :math:`Q` is nonzero and vice-versa) and ` (:math:`P(x)=Q(x) \\forall x`)

    Parameters
    --------------
    kde1, kde2: sklearn KDE models
        KDEs for distributions to compare.

    samples: int
        number of evenly-spaced points on the intevral :math:`[0,1]`

    """
    # Compute the PDFs of the two KDEs at some common evaluation points
    # How likely each data point is according to the KDE model. Quantifies how well each data point fits the estimated probability distribution.
    x = np.linspace(0, 1, samples)
    pdf_kde1 = _kde_to_pdf(kde1, x)
    pdf_kde2 = _kde_to_pdf(kde2, x)


    squared_diff = (np.sqrt(pdf_kde1)-np.sqrt(pdf_kde2))**2
    area = trapezoid(squared_diff, x)
    d_hellinger = np.sqrt(area/2)

    return 1 - d_hellinger


def kl_distance(kde1, kde2, samples: int):
    # Compute the PDFs of the two KDEs at some common evaluation points
    # How likely each data point is according to the KDE model. Quantifies how well each data point fits the estimated probability distribution.
    x = np.linspace(0, 1, samples)
    pdf_kde1 = _kde_to_pdf(kde1, x)
    pdf_kde2 = _kde_to_pdf(kde2, x)

    # Compute the Kullback-Leibler Distance using samples
    kl = entropy(pdf_kde1, pdf_kde2)
    # TODO note - KL is not bounded between 0 and 1- inverting may give negative values
    return 1 - kl

# Jensen-Shannon Divergence
def js_distance(kde1, kde2, samples: int):
    # Compute the PDFs of the two KDEs at some common evaluation points
    # How likely each data point is according to the KDE model. Quantifies how well each data point fits the estimated probability distribution.
    x = np.linspace(0, 1, samples)
    pdf_kde1 = _kde_to_pdf(kde1, x)
    pdf_kde2 = _kde_to_pdf(kde2, x)

    if np.allclose(pdf_kde1, pdf_kde2):
        # If two kdes are functionally identical but off by a 10 to the minus 6 or so floating point amount
        # jensenshannon can hit floating point roundoff problems and return a nan instead of a zero.
        # To avoid introducing nans by hitting this case, we'll set very close to zero cases to zero.
        js = 0.0
    else:
        # Compute the Jensen-Shannon Distance using samples
        js = jensenshannon(pdf_kde1, pdf_kde2)

    # 1 = unaligned, 0 = full aligned
    return js
