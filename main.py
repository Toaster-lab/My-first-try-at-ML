import random as r

strings = [
    list('asytbtsd'),
    list('asyhbtsd'),
    list('tshgguvh'),
    list('asytxasd'),
    list('asytbtsd'),
    list('asytbqsd'),
    list('asytbasd'),
    list('xsyfbtsd'),
    list('asygbtsd'),
    list('asytbtsq'),
    list('zsyzbzsd'),
    list('asytxbsd'),
    list('asytbqsd'),
    list('qwertyui'),
    list('asdfghjk'),
    list('zxcvbnmq'),
    list('asytbtad'),
    list('asyabtsd'),
    list('asytbtsf'),
    list('asytbtsz'),
    list('psytbtsd'),
    list('asyxbtsd')
]

original = 'asytbtsd'

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


def calculating_odds():
    score = scores()

    for i in range(len(odds)):
        odds[i] = (
            score[i] * [1]
            + (len(original) - score[i]) * [0] + [2])
    return odds

def ranking():
    scores()

    ranking = sorted(
        range(len(score)),
        key=lambda i: score[i],
        reverse=True
    )
    best_score = score[ranking[0]]
    best_string = strings[ranking[0]]
    second_score = score[ranking[1]]
    second_string = strings[ranking[1]]
    chances = chances = best_score**2 *[best_string] + second_score**2 *[second_string]
    print(chances)


ranking()