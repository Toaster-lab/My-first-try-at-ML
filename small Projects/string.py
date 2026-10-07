import random as r

strings = [
    list('asdfbasd'), list('asyfbtsd'), list('tshgguvh'), list('asytxasd'),
    list('qsyttbsd'), list('asytbqsd'), list('asytbasd'), list('xsyfbtsd'),
    list('asygbtsd'), list('asytbtsq'), list('zsyzbzsd'), list('asytxbsd'),
    list('asytbqsd'), list('qwertyui'), list('asdfghjk'), list('zxcvbnmq'),
    list('asytbtad'), list('asyabtsd'), list('asytbtsf'), list('asytbtsz'),
    list('psytbtsd'), list('asyxbtsd')
]

original = 'asytbtsd'

mutation = list(map(chr, range(97, 123)))

score = [0] * len(strings)
odds = [4] * len(strings)

def scores():
    global score
    score = [0] * len(strings)
    for j in range(len(strings)):
        for i in range(len(original)):
            if strings[j][i] == original[i]:
                score[j] += 1
    return score


def breading():
    global strings
    odds = calculating_odds()
    for j in range(len(strings)):
        for i in range(len(original)):
            choice : int = r.choice(odds[j]) #i could change the odds by adding more 1s or more 0s
            if choice == 1:
                if j == len(strings) - 1:
                    strings[j][i] = strings[0][i]
                else:
                    strings[j][i] = strings[j+1][i]
            elif choice == 2:
                strings[j][i] = r.choice(mutation)


def calculating_odds():
    score = scores()
    for i in range(len(odds)):
        odds[i] = score[i]*[1] + (len(original)-score[i])*[0] + [2]
    return odds

def main():
    found_it : bool = False
    while True:
        breading()
        for j in range(len(strings)):
            compare = 0
            for i in range(len(original)):
                if strings[j][i] != original[i]:
                    break
                else:
                    compare += 1
            if compare == len(original):
                found_it = True
                print("".join(strings[j]))
                break
        if found_it:
            break
main()
