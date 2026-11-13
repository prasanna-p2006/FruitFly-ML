import sys
import os

# Ensure src directory is in sys.path so cs229 submodules can be imported cleanly
_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

