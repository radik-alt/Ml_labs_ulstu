from labs.five.Five import Five
from labs.four.Four_lab import Four_lab
from labs.third.Third import Third
from labs.first.First import First
from labs.second.Second import Second

if __name__ == "__main__":
    n = 3
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
