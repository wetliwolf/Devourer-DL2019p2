# ========================================================================
# Notebook 0
# ========================================================================

TEST = 'test'

# ========================================================================
# Notebook 1
# ========================================================================

import operator
def test(a,b,cmp,cname=None):
    if cname is None:
        cname=cmp.__name__
    assert cmp(a,b),f"{cname}:\n{a}\n{b}"

def test_eq(a,b):
    test(a,b,operator.eq,'==')

from pathlib import Path
from IPython.core.debugger