import os
from six.moves import urllib
import tarfile
import zipfile
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def download_files(desired_file, download_url, fn, cache_folder=os.getcwd(),
                   force=False):
    if not os.path.isfile(desired_file):
        dest = os.path.join(cache_folder, fn)
        if not os.path.isfile(os.path.join(cache_folder, fn)) and not force:
            urllib.request.urlretrieve(download_url, dest)

        # extract tarball
        if (dest.endswith("tar.gz")):
            tar = tarfile.open(dest, "r:gz")
            tar.extractall()
            tar.close()
        elif dest.endswith('.zip'):
            zip_ref = zipfile.ZipFile(dest, 'r')
            zip_ref.extractall()
            zip_ref.close()
    return desired_file


def execute_notebook(path_to_notebook):
    with open(path_to_notebook) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {
            'path': os.path.dirname(path_to_notebook)}})
