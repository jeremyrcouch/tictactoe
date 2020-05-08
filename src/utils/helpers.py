import numpy as np


# not currently needed, keeping for reference
def array_in_list(arr, arr_list):
   return next((True for elem in arr_list if np.array_equal(elem, arr)), False)