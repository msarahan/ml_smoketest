import os
import utils
import shutil
import subprocess


def test_cnn():
    notebook_filename = 'caffe-1.0/examples/net_surgery.ipynb'
    utils.download_files(desired_file=notebook_filename,
                         download_url="https://github.com/BVLC/caffe/archive/1.0.tar.gz",
                         fn="caffe-1.0.tar.gz")
    # net_surgery is one of their examples that we've hacked to be more friendly to our setup
    shutil.copy2(os.path.join(os.path.dirname(__file__), 'net_surgery.py'),
                 os.path.dirname(notebook_filename))
    subprocess.check_call(['python', 'net_surgery.py'], cwd=os.path.dirname(notebook_filename))
