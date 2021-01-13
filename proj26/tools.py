from torch import sparse


def sparse_dot(m1, m2):
    """"
        Sparse dot product
    """
    y = sparse.sum(m1 * m2)
    return y

def sparse_argmax(m1):
    argmax = m1.values.argmax()
