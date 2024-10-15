import timeit


def time_wrapper(func):
    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        result = func(*args, **kwargs)
        end = timeit.default_timer()
        print(f"{func.__name__} 실행 시간 : {end - start:.2f}초")
        return result
    return wrapper


if __name__ == "__main__":
    import time

    @time_wrapper
    def test():
        time.sleep(2)
        return "Hello"
    
    print(test())