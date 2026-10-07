import random as r

strings = [
    list('asdfbasd'), list('asyfbtsd'), list('tshgguvh'), list('asytxasd'),
    list('qsyttbsd'), list('asytbqsd'), list('asytbasd'), list('xsyfbtsd'),
    list('asygbtsd'), list('asytbtsy'), list('zsyzbzsd'), list('asytxbsd'),
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
    pool = calculating_pool()
    for j in range(len(strings)):
        for i in range(len(original)):
            choice = r.choice(pool) 
            if choice == 2:
                strings[j][i] = r.choice(mutation)
            else:
                strings[j][i] = choice[i]


''' Used in old version: def calculating_odds():
    score = scores()
    for i in range(len(odds)):
        odds[i] = score[i]*[1] + (len(original)-score[i])*[0] + [2]
    return odds '''

def ranking():
    score = scores()
    ranking = sorted(range(len(score)), key=lambda i: score[i], reverse=True)
    best_score = score[ranking[0]]
    best_string = strings[ranking[0]]
    second_score = score[ranking[1]]
    second_string = strings[ranking[1]]
    return best_score, best_string, second_score, second_string

def calculating_pool():
    best_score, best_string, second_score, second_string = ranking()
    pool = best_score*2 *[best_string] + second_score*1 *[second_string] + [2]
    return pool

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
