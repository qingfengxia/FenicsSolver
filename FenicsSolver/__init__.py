__version__ = "0.1"
__author__ = 'Qingfeng Xia'

#__all__ = []

# a function can be execute here to enable
# python3 -m FenicsSolver config.json

import sys
from .main import main

if len(sys.argv) >= 2:
    main(sys.argv)