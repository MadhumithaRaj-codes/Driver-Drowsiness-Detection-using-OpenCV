from scipy.spatial import distance as dist

def mouth_aspect_ratio(mouth):
    # vertical distances
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])

    # horizontal distance
    D = dist.euclidean(mouth[12], mouth[16])

    # MAR formula
    mar = (A + B + C) / (2.0 * D)
    return mar
