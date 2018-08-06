# https://github.com/ContinuumIO/anaconda-issues/issues/8803#issuecomment-371552379
def test_linalg():
    import numpy as np
    import scipy.linalg

    n = 1024
    eye = np.eye(n+1, n+1)

    u, s, vh = scipy.linalg.svd(eye)
    eye2 = u.dot(np.diag(s)).dot(vh)

    all_close = np.allclose(eye, eye2)
    assert all_close
