def iter():
    for i in range(10):
        yield i*i

if __name__ == "__main__":
    x = iter()
    print(next(x))
    print(next(x))
    print(list(x))
