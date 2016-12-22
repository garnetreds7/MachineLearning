import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kw):
        tic = time.clock()
        result = func(*args, **kw)
        toc = time.clock()
        print 'function %s costs time: %f seconds' %(func.__name__, (toc-tic))
        return result
    return wrapper