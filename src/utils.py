""" Description here

Author: Leonard Berrada
Date: 23 Oct 2015
"""

import time


def timeit(my_func):
    """time the execution of the function given in argument
    
    :param: my_func (func) : function to be timed
    :print: execution time of my_func
    :return: timed (my_func result-like) : result of my_func
    """

    def timed(*args, **kw):
        """get result of my_func with given arguments
        
        :param: *args (any argument) : packed unnamed arguments to pass to my_func
        :param: *kw (any argument) : packed named arguments to pass to my_func
        :return: result (my_func result-like) : result of my_func with gven arguments
        """
        
        ts = time.time()
        result = my_func(*args, **kw)
        te = time.time()

        print('%r took %2.3g sec' %
              (my_func.__name__, te - ts))
        return result

    return timed