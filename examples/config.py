import sys
import os
import os.path

# make test_*.py work for developer, without install this package
try:
    import FenicsSolver
except:
    parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
    sys.path.append(parent_dir)


def is_interactive():
    return not is_batch()

# run all the test under example folder: pytest BATCH=1
def is_batch():
    if 'pytest' in sys.argv:
        return True
    if 'BATCH' in os.environ:
        return True
    if os.environ.get('FENICSSOLVER_BATCH', False):  # an env var set in run_all_tests.py
        return True
    return False