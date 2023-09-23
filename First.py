import re
import pandas as pd
from pandas import DataFrame


class First:

    def __int__(self):
        print()

    def parse_file(self):
        data = pd.read_csv('titanic_train.csv')
        self.first_lab(data)

    def first_lab(self, data: DataFrame):
        print("1.Сколько мужчин / женщин было на борту?")
        print(data.groupby(["Sex"])["Sex"].value_counts())

        print(
            "\n2.Определите распределение функции Pclass. Теперь Для мужчин и женщин отдельно. Сколько людей из второго класса было на борту?")
        people = data[data["Pclass"] == 2]
        print(len(people))

        print("\n3.Каковы медиана и стандартное отклонение Fare?. Округлите до 2-х знаков после запятой.")
        median_fare = round(data["Fare"].median(), 2)
        std_deviation_fare = round(data["Fare"].std(), 2)

        print("Медиана Fare:", median_fare)
        print("Стандартное отклонение Fare:", std_deviation_fare)

        print(
            "\n4.Правда ли, что средний возраст выживших людей выше, чем у пассажиров, которые в конечном итоге умерли?")
        survived = data[data["Survived"] == 1]
        dead = data[data["Survived"] == 0]

        mean_age_survived = survived["Age"].mean()
        mean_age_dead = dead["Age"].mean()

        if mean_age_survived > mean_age_dead:
            print("ДА")
        else:
            print("НЕТ")

        print(
            "\n5.Это правда, что пассажиры моложе 30 лет. выжили чаще, чем те, кому больше 60 лет. Каковы доли выживших людей среди молодых и пожилых людей?")
        young = data[data["Age"] <= 30]
        elderly = data[data["Age"] >= 60]

        survival_rate_young = young["Survived"].mean()
        survival_rate_elderly = elderly["Survived"].mean()

        print("Доля выживших среди молодых пассажиров:", round(survival_rate_young * 100, 2))
        print("Доля выживших среди пожилых пассажиров:", round(survival_rate_elderly * 100, 2))

        print("\n6.Правда ли, что женщины выживали чаще мужчин? Каковы доли выживших людей среди мужчин и женщин?")
        female = data[data["Sex"] == "female"]
        male = data[data["Sex"] == "male"]

        survival_rate_female = female["Survived"].mean()
        survival_rate_male = male["Survived"].mean()

        print("Доля выживших среди женщин:", round(survival_rate_female * 100, 2))
        print("Доля выживших среди мужчин:", round(survival_rate_male * 100, 2))

        print("\n7.Какое имя наиболее популярно среди пассажиров мужского пола?")
        male_passengers = data[data["Sex"] == "male"]

        male_names = male_passengers["Name"]

        all_male = []
        for name in male_names:

            full_name = re.findall(r'(?:Mr\.|Mrs\.)\s+([A-Z][a-z]+)', name)
            if len(full_name) == 0:
                full_name = re.findall(r'Master.\s+([A-Z][a-z]+)', name)

            if len(full_name) == 0:
                full_name = re.findall(r',\s*([A-Za-z\s]+)', name)

            if len(full_name) != 0:
                first_name = full_name[0]
                all_male.append(first_name)

        name_counts = pd.Series(all_male).value_counts().sort_values(ascending=True)
        most_popular_name = name_counts.idxmax()
        print("Наиболее популярное имя среди пассажиров мужского пола:", most_popular_name)

        print("\n8.Как средний возраст мужчин / женщин зависит от Pclass? Выберите все правильные утверждения:")

        male_passengers = data[data["Sex"] == "male"]
        female_passengers = data[data["Sex"] == "female"]

        average_age_male = male_passengers.groupby("Pclass")["Age"].mean()
        average_age_female = female_passengers.groupby("Pclass")["Age"].mean()

        condition1 = average_age_male[1] > 40
        condition2 = average_age_female[1] > 40
        condition3 = (average_age_male > average_age_female).all()
        condition4 = (average_age_female[1] > average_age_female[2]) and (average_age_female[2] > average_age_female[3])

        print("1. В среднем мужчины 1 класса старше 40 лет:", condition1)
        print("2. В среднем женщины 1 класса старше 40 лет:", condition2)
        print("3. Мужчины всех классов в среднем старше, чем женщины того же класса:", condition3)
        print(
            "4. В среднем, пассажиры первого класса старше, чем пассажиры 2-го класса, которые старше, чем пассажиры 3-го класса:",
            condition4)


