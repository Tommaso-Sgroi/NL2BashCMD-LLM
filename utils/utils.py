from time import time as t, sleep as s



def timeit(func):
    def time(*args, **kwargs):
        start = t()
        r = func(*args, **kwargs)
        end = t()
        return end - start, r
    return time

