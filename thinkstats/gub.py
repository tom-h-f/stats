#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import empiricaldist

from thinkstats import decorate


from empiricaldist import FreqTab

ftab = FreqTab.from_seq([1, 2, 2, 3, 5])
print(ftab)
