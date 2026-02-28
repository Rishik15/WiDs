import numpy as np


def cindex(n, probs, time_to_hit, event):
    p12 = probs[:, 0]
    p24 = probs[:, 1]
    p48 = probs[:, 2]
    p72 = probs[:, 3]

    q12 = p12
    q24 = p24 - p12
    q48 = p48 - p24
    q72 = p72 - p48
    q_after = 1 - p72

    t_hat = q12 * 6 + q24 * 18 + q48 * 36 + q72 * 60 + q_after * 84

    risk = -t_hat

    concordant = 0
    ties = 0
    comparable = 0

    for i in range(n):
        for j in range(i + 1, n):

            if time_to_hit[i] == time_to_hit[j]:
                continue

            if time_to_hit[i] < time_to_hit[j]:
                if event[i] == 0:
                    continue
                comparable += 1
                if risk[i] > risk[j]:
                    concordant += 1
                elif risk[i] == risk[j]:
                    ties += 1

            else:
                if event[j] == 0:
                    continue
                comparable += 1
                if risk[j] > risk[i]:
                    concordant += 1
                elif risk[j] == risk[i]:
                    ties += 1

    c_index = (concordant + 0.5 * ties) / comparable if comparable > 0 else 0.5

    return c_index


def weighted_brier(n, probs, event, time_to_hit):
    horizons = [24, 48, 72]
    horizon_cols = [1, 2, 3]

    briers = []

    for H, col in zip(horizons, horizon_cols):

        p = probs[:, col]

        mask = []
        y = []

        for i in range(n):

            if event[i] == 1 and time_to_hit[i] <= H:
                mask.append(True)
                y.append(1)

            elif time_to_hit[i] > H:
                mask.append(True)
                y.append(0)

            else:
                mask.append(False)
                y.append(0)

        mask = np.array(mask)
        y = np.array(y)

        if mask.sum() == 0:
            briers.append(0.0)
        else:
            brier = np.mean((p[mask] - y[mask]) ** 2)
            briers.append(brier)

    brier_24, brier_48, brier_72 = briers

    weighted_brier = 0.3 * brier_24 + 0.4 * brier_48 + 0.3 * brier_72

    return weighted_brier


def metric(probs, time_to_hit, event):
    probs = np.asarray(probs)
    time_to_hit = np.asarray(time_to_hit)
    event = np.asarray(event)

    n = len(time_to_hit)

    c_index = cindex(n, probs, time_to_hit, event)

    brierscore = weighted_brier(n, probs, event, time_to_hit)

    final_score = 0.3 * c_index + 0.7 * (1 - brierscore)

    return c_index, brierscore, final_score
