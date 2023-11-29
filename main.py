from Five import Five
from Four_lab import Four_lab
from Third import Third
from First import First
from Second import Second

if __name__ == "__main__":
    n = 4
    if n == 1:
        First().parse_file()
    elif n == 2:
        Second().second_lab()
    elif n == 3:
        Third()
    elif n == 4:
        Four_lab()
    elif n == 5:
        Five()
    else:
        print("Такой лабы нет!")
