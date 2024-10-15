import time
import timeit

def time_wrapper(func):
    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        result = func(*args, **kwargs)
        end = timeit.default_timer()
        print(f"{func.__name__} 실행 시간 : {end - start:.2f}초")
        return result
    return wrapper

def get_current_time():
    return time.strftime("%Y-%m-%d_%H-%M-%S")

if __name__ == "__main__":
    @time_wrapper
    def test():
        time.sleep(0.1)
        return "Hello"
    
    print(test())
    print(get_current_time())