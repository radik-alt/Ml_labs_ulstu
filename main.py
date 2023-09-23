from First import First
from Second import Second

if __name__ == "__main__":
    print("Выберите номер лаб.работы:")
    n = int(input())
    if n == 1:
        First().parse_file()
    elif n == 2:
        Second().second_lab()
    else:
        print("Такой лабы нет!")

