import timeit


def time_wrapper(func):
    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        result = func(*args, **kwargs)
        end = timeit.default_timer()
        print(f"Elapsed time: {end - start}")
        return result
    return wrapper


if __name__ == "__main__":
    import time

    @time_wrapper
    def test():
        time.sleep(2)
        return "Hello"
    
    print(test())