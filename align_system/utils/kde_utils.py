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


def load_kde(target_kdma, norm='globalnorm'):
    if isinstance(target_kdma, KDMAValue):
        target_kdma = target_kdma.to_dict()

    if norm == 'globalnorm':
        target_kde = kde_from_base64(target_kdma['kdes']['globalnorm']['kde'])
    elif norm == 'localnorm':
        target_kde = kde_from_base64(target_kdma['kdes']['localnorm']['kde'])
    elif norm == 'rawscores':
        target_kde = kde_from_base64(target_kdma['kdes']['rawscores']['kde'])
    elif norm == 'globalnormx_localnormy':
        target_kde = kde_from_base64(target_kdma['kdes']['globalnormx_localnormy']['kde'])
    elif norm == 'priornorm':
        norm_factor = 0.3
        # Load KDE
        linspace = np.linspace(0, 1, 1000)
        kde = kde_from_base64(target_kdma['kdes']['rawscores']['kde'])
        density = _kde_to_pdf(kde, linspace)

        # Get prior KDE
        prior_data = [0.1]*325 + [0.3]*60 + [0.7]*30 + [0.9]*300
        prior_kde = get_kde_from_samples(prior_data)
        prior_density = _kde_to_pdf(prior_kde, linspace)

        # Normalize the target KDE based on prior KDE
        normalized_density = density / (prior_density + 1e-10)
        # Weight the normalization
        normalized_density = (norm_factor*normalized_density) + ((1-norm_factor)*density)

        # Use normalized density to construct KDE
        norm_samples = []
        for value in [0.1, 0.3, 0.7, 0.9]:
            norm_samples += [value]*round(normalized_density[int(value*len(linspace))]*100)
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

def get_kde_from_samples(X: list[float], norm='rawscores'):
    """
    Generates a KDE based on a sample X
    """
    X = np.array(X) # convert to numpy (if not already)
    if norm == 'globalnormx_localnormy':
        bandwidth = get_default_2feature_bandwidth()
        kde = make_2feature_kde(X, X, bandwidth)
    else:
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


# https://github.com/ITM-Soartech/itm-analysis/blob/develop/src/itm/program_metrics_pipeline/fingerprint_library.py#L196
def js_distance_2d(kde1, kde2, grid_size=100):

    # Compute the PDFs of the two 3D KDEs on the 2D grid
    pdf_kde1 = _kde_to_pdf_2feature(kde1, grid_size)
    pdf_kde2 = _kde_to_pdf_2feature(kde2, grid_size)

    if np.allclose(pdf_kde1, pdf_kde2):
        # If two kdes are functionally identical but off by a 10 to the minus 6 or so floating point amount
        # jensenshannon can hit floating point roundoff problems and return a nan instead of a zero.
        # This is not a 2D kde issue, it can happen in 1D as well.
        # To see it happen in a simple 1D case these values will cause the error case
        # if they are used as the input to fitting two kdes to compare with this function
        # (6 zeros and 6 1s in each array, but not at the same array positions)
        #kde1_data_values = np.array([1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0])
        #kde2_data_values = np.array([0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0])
        # To avoid introducing nans by hitting this case, we'll set very close to zero cases to zero.
        print("KDE1 and KDE2 are so close to identical Jensenshannon can produce a nan. Assigning them to be identical.")
        js = 0.0
    else:
        js = jensenshannon(pdf_kde1, pdf_kde2)

    # We invert the value because the spec agreed to with the other ITM performs has
    # 0 = unaligned, 1 = full aligned which is the opposite of what Jensenshannon produces.
    return 1 - js

# https://github.com/ITM-Soartech/itm-analysis/blob/develop/src/itm/program_metrics_pipeline/fingerprint_library.py#L174S
def _kde_to_pdf_2feature(kde, grid_size=100, normalize=True):

    # Create a 2D grid of points within the normalized range [0, 1] x [0, 1]
    x_vals = np.linspace(0, 1, grid_size)
    y_vals = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x_vals, y_vals)
    xy_grid = np.vstack([X.ravel(), Y.ravel()]).T

    # Evaluate the KDE on the xy_grid
    pf = np.exp(kde.score_samples(xy_grid))

    if normalize:
        # Reshape the grid and the probability values
        x_vals = np.linspace(0, 1, grid_size)
        y_vals = np.linspace(0, 1, grid_size)
        pf_reshaped = pf.reshape(grid_size, grid_size)

        # Normalize the 2D KDE values
        pf = _normalize_2feature(x_vals, y_vals, pf_reshaped).ravel()

    return pf

# https://github.com/ITM-Soartech/itm-analysis/blob/develop/src/itm/program_metrics_pipeline/fingerprint_library.py#L142
def _normalize_2feature(x, y, z):
    """
    Normalize 2D probability distribution z such that its integral over domain (x, y) is one.

    Parameters
    ----------
    x: ndarray
        domain over which discrete probability distribution z is defined (x-coordinates).

    y: ndarray
        domain over which discrete probability distribution z is defined (y-coordinates).

    z: ndarray
        2D probability distribution at each point in (x, y). z is proportional to the
        probability density of the distribution at (x, y).

    Returns
    --------
    pdf: ndarray
        array with same shape as z that gives normalized probability density function
        values at each point (x, y).
    """
    # Compute the area under the surface
    dx = x[1] - x[0]  # assuming uniform spacing
    dy = y[1] - y[0]  # assuming uniform spacing
    area = trapezoid(trapezoid(z, x, axis=0), y, axis=0)

    # Normalize z by the computed area
    pdf = z / area

    return pdf

# https://github.com/ITM-Soartech/itm-analysis/blob/develop/src/itm/program_metrics_pipeline/fingerprint_library.py#L33
def make_2feature_kde(X: list[float], Y: list[float], bandwidth, max_value=1.0):
    # Convert input lists to numpy arrays
    X = np.array(X)
    Y = np.array(Y)

    # Concatenate X and Y to form a 2D array where each row is (X[i], Y[i])
    data = np.column_stack((X, Y))

    # Fit Kernel Density Estimation
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(data)

    return kde

# https://github.com/ITM-Soartech/itm-analysis/blob/develop/src/itm/program_metrics_pipeline/fingerprint_library.py#L48
def get_default_2feature_bandwidth(max_value=1.0):
    bandwidth = (max_value / 10) * 0.75
    return bandwidth