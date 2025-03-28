def getOneBits(n: int) -> list[int]:
    """
    Computes:
    - Number of 1-bits in binary representation of n
    - Positions (1-indexed, left to right) of each 1-bit

    Assumes: 1 < n < 10^9
    """
    if not (1 < n < 10**9):
        raise ValueError("Input n must satisfy: 1 < n < 10^9")

    k = n.bit_length()
    result = []
    for i in range(k):
        if (n >> (k - i - 1)) & 1:
            result.append(i + 1)
    return [len(result)] + result


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = getOneBits(n)
