"""Shared constants used by the experimental benchmark tools.

Changing these values changes the evaluation protocol. Any released result
should therefore record the exact values used.
"""

NUMERIC_TOL_PCT = 2
GEO_DIFF_REL_TOL = 0.01

# A SHA-1 final byte below 51 assigns a program to the approximate 20% test split.
SPLIT_TEST_THRESHOLD = 51
