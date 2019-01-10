import sys
import os

def is_interactive():
    return not is_batch()

# run all the test under example folder: pytest BATCH=1
def is_batch():
    if 'pytest' in sys.argv:
        return True
    if 'BATCH' in os.environ:
        return True
    return False