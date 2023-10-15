import numpy

import TestMl
from Third import Third
from First import First
from Second import Second

if __name__ == "__main__":
    print("Выберите номер лаб.работы:")
    n = 3
    if n == 1:
        First().parse_file()
    elif n == 2:
        Second().second_lab()
    elif n == 3:
        Third()
    else:
        TestMl.MyWork()
        # print("Такой лабы нет!")
