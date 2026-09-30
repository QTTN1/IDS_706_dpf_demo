# Makes sure sales_analysis.py is importable from tests/, regardless of what directory pytest is invoked from.
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
