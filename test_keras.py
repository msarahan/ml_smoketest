import os
import sys
import utils


def test_optimizers():
    fn = 'keras-master/tests/keras/optimizers_test.py'
    archive_file = "keras_master.zip"
    utils.download_files(fn, "https://github.com/keras-team/keras/archive/master.zip", archive_file)
    sys.path.append(os.path.join(os.getcwd(), 'keras-master/tests/keras'))

    import optimizers_test
    optimizers_test.test_sgd()
    optimizers_test.test_adagrad()
    optimizers_test.test_no_grad()
    optimizers_test.test_tfoptimizer()
