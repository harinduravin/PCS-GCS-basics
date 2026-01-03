import numpy as np

def n_choose_k_iter_log(n, k):
    """
    Calculate log2 of binomial coefficient (n choose k)
    """
    i = np.arange(1, k + 1)
    return np.sum(np.log2((n - (k - i)) / i))


def n_choose_ks_recursive_log2(n, k_vec):
    """
    Calculates log2(n! / prod_i k(i)!)
    n: scalar
    k_vec: array-like, sum(k_vec) = n
    """
    out = 0.0
    k = np.sort(np.array(k_vec))
    for i in range(len(k) - 1):
        out += n_choose_k_iter_log(n, k[i])
        n -= k[i]
    return out


def idquant(p, n):
    """
    Quantize to n-type distribution using iterative procedure
    """
    m = len(p)
    n_i = np.zeros(m, dtype=int)
    t = np.log(1.0 / p)
    p_quant = t.copy()

    # iterative quantization loop
    for _ in range(n):
        index = np.argmin(p_quant)
        cj = n_i[index] + 1
        n_i[index] = cj
        p_quant[index] = (cj + 1) * np.log(cj + 1) - cj * np.log(cj) + t[index]

    p_quant = n_i / n
    return n_i, p_quant


def initialize(p, n):
    """
    Quantize to n-type distribution and compute number of info bits.
    """
    n_i, p_quant = idquant(p, n)
    num_info_bits = np.floor(n_choose_ks_recursive_log2(n, n_i))
    return p_quant, int(num_info_bits), n_i
