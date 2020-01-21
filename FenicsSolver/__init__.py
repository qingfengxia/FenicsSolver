__version__ = "0.1"
__author__ = 'Qingfeng Xia'

#__all__ = []

# a function can be execute here to enable
# python3 -m FenicsSolver config.json

import sys
from .main import main

#print(sys.argv)
if len(sys.argv) < 2:
    print("Not enough input argument, Usage: `python - FenicsSolver case_input` \n  to run simulation")
    #  must start this solver in FenicsSolver folder

else:
    config_file = sys.argv[1]
    print("run FenicsSolver with config file", config_file)
    main(config_file)