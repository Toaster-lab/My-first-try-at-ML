import random as r

strings = [
    list('asdfbasd'),
    list('asyfbtsd'),
    list('tshgguvh')]

score = [0,
         0]

original = 'asytbtsd'


def scores():
    for j in range(len(strings)):
        for i in range(len(original)):
            if strings[j][i] == original[i]:
                score[j] += 1


def breading():
    for j in range(len(strings)):
        for i in range(len(original)):
            choice : int = r.choice([0,1])
            if choice == 1:
                if j == len(strings) - 1:
                    strings[j][i] = strings[0][i]
                else:
                    strings[j][i] = strings[j+1][i]


breading()
strings = ["".join(s) for s in strings]
print(strings)
