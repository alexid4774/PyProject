'''This is a file with the implementation of NIST tests in the form of functions.'''

from collections import Counter
from math import erfc, exp, fabs, floor, gamma, lgamma, log, pi, sqrt, cos, sin
from typing import Iterable, Sequence
from scipy.stats import norm
import numpy as np

input_types = str | Sequence[int] | Iterable[int]


def _to_bits(bits: input_types) -> list[int]:
    
    '''Converts a bit string or iterable of 0/1 values into a validated list of ints.'''
    
    if isinstance(bits, str):
        result = [int(ch) for ch in bits.strip() if ch in "01"]
        if len(result) != len(bits.strip()):
            raise ValueError("String input must contain only '0' and '1'.")
        return result

    result = [int(x) for x in bits]
    if any(x not in (0, 1) for x in result):
        raise ValueError("Input must contain only 0 and 1 values.")
    return result


def _igamc(a: float, x: float) -> float:
    
    '''Approximates the complemented incomplete gamma function Q(a, x).'''
    
    if x <= 0:
        return 1
    if a <= 0:
        raise ValueError("Parameter 'a' must be positive.")

    eps = 1e-14
    max_iter = 200
    gln = lgamma(a)

    if x < a + 1:
        term = 1 / a
        total = term
        ap = a
        for _ in range(max_iter):
            ap += 1
            term *= x / ap
            total += term
            if abs(term) < abs(total) * eps:
                break
        p = total * exp(-x + a * log(x) - gln)
        return max(0, min(1, 1 - p))

    b = x + 1 - a
    c = 1 / 1e-300
    d = 1 / b
    h = d
    
    for i in range(1, max_iter + 1):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < eps:
            break
    q = exp(-x + a * log(x) - gln) * h
    
    return max(0, min(1, q))


def _rank_binary_matrix(matrix: list[list[int]]) -> int:
    
    '''Computes rank of a binary matrix over GF(2) using Gaussian elimination.'''
    
    rows = [row[:] for row in matrix]
    row_count = len(rows)
    col_count = len(rows[0]) if rows else 0
    rank = 0

    for col in range(col_count):
        pivot = None
        for r in range(rank, row_count):
            if rows[r][col] == 1:
                pivot = r
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for r in range(row_count):
            if r != rank and rows[r][col] == 1:
                rows[r] = [a ^ b for a, b in zip(rows[r], rows[rank])]
        rank += 1
        if rank == row_count:
            break

    return rank


def frequency_monobit_test(bits: input_types) -> float:
    
    '''Checks whether the number of ones and zeros is approximately balanced.'''
    
    data = _to_bits(bits)
    n = len(data)
    
    if n == 0:
        raise ValueError("Input sequence must not be empty.")
    
    s_obs = abs(sum(1 if bit else -1 for bit in data)) / sqrt(n)
    
    return erfc(s_obs / sqrt(2))


def frequency_block_test(bits: input_types, block_size: int = 128) -> float:
    
    '''Splits the sequence into blocks and checks balance of ones inside each block.'''
    
    data = _to_bits(bits)
    n = len(data)
    
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    
    block_count = n // block_size
    
    if block_count == 0:
        raise ValueError("Input is too short for the selected block_size.")

    chi_square = 0
    
    for i in range(block_count):
        block = data[i * block_size:(i + 1) * block_size]
        proportion = sum(block) / block_size
        chi_square += (proportion - 0.5) ** 2
    chi_square *= 4 * block_size
    
    return _igamc(block_count / 2, chi_square / 2)


def runs_test(bits: input_types) -> float:
    
    '''Checks whether runs of equal bits occur too often or too rarely.'''
    
    data = _to_bits(bits)
    n = len(data)
    
    if n < 2:
        raise ValueError("Input sequence must contain at least two bits.")
    
    pi_hat = sum(data) / n
    if abs(pi_hat - 0.5) >= 2 / sqrt(n):
        return 0
    
    runs = 1 + sum(data[i] != data[i - 1] for i in range(1, n))
    numerator = abs(runs - 2 * n * pi_hat * (1 - pi_hat))
    denominator = 2 * sqrt(2 * n) * pi_hat * (1 - pi_hat)
    
    return erfc(numerator / denominator)


def longest_run_ones_test(bits: input_types, block_size: int = 128) -> float:

    '''Checks whether the longest run of ones in each block matches expectation.'''

    data = _to_bits(bits)
    n = len(data)

    if block_size not in (8, 128, 10_000):
        raise ValueError("block_size must be one of 8, 128, or 10000.")

    block_count = n // block_size

    if block_count == 0:
        raise ValueError("Input is too short for the selected block_size.")

    if block_size == 8:
        thresholds = [1, 2, 3, 4]
        probabilities = [0.2148, 0.3672, 0.2305, 0.1875]
    elif block_size == 128:
        thresholds = [4, 5, 6, 7, 8, 9]
        probabilities = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
    else:
        thresholds = [10, 11, 12, 13, 14, 15, 16]
        probabilities = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]

    counts = [0] * len(probabilities)
    for i in range(block_count):
        block = data[i * block_size:(i + 1) * block_size]
        longest = max((len(run) for run in ''.join(map(str, block)).split('0')), default=0)
        index = 0
        while index < len(thresholds) - 1 and longest > thresholds[index]:
            index += 1
        counts[index] += 1

    chi_square = sum((counts[i] - block_count * probabilities[i]) ** 2 / (block_count * probabilities[i]) for i in range(len(counts)))

    return _igamc((len(probabilities) - 1) / 2, chi_square / 2)


def binary_matrix_rank_test(bits: input_types, rows: int = 32, columns: int = 32) -> float:

    '''Builds binary matrices and checks whether their ranks follow expected frequencies.'''

    data = _to_bits(bits)
    matrix_size = rows * columns
    matrix_count = len(data) // matrix_size

    if matrix_count == 0:
        raise ValueError("Input is too short for one complete matrix.")
    if rows != 32 or columns != 32:
        raise ValueError("This simplified implementation supports only 32x32 matrices.")

    full_rank = 0
    rank_31 = 0
    lower_rank = 0

    for i in range(matrix_count):
        chunk = data[i * matrix_size:(i + 1) * matrix_size]
        matrix = [chunk[r * columns:(r + 1) * columns] for r in range(rows)]
        rank = _rank_binary_matrix(matrix)
        if rank == 32:
            full_rank += 1
        elif rank == 31:
            rank_31 += 1
        else:
            lower_rank += 1

    probabilities = [0.2888, 0.5776, 0.1336]
    observed = [full_rank, rank_31, lower_rank]
    chi_square = sum((observed[i] - matrix_count * probabilities[i]) ** 2 / (matrix_count * probabilities[i]) for i in range(3))

    return exp(-chi_square / 2)


def discrete_fourier_transform_test(bits: input_types) -> float:

    '''Uses a direct Fourier transform to detect periodic patterns in the sequence.'''

    b = np.asarray(bits, dtype = int)
    n = len(b)

    x = 2 * b - 1

    spectrum = np.abs(np.fft.fft(x))[:n // 2]

    threshold = sqrt(n * log(1 / 0.05))
    expected = 0.95 * n / 2
    observed = np.sum(spectrum < threshold)

    variance = n * 0.95 * 0.05 / 4.0
    d = (observed - expected) / sqrt(variance)

    return erfc(abs(d) / sqrt(2))


def non_overlapping_template_test(bits: input_types, template: str = "001", block_size: int = 1032) -> float:

    '''Counts non-overlapping occurrences of a chosen template inside fixed-size blocks.'''

    data = ''.join(map(str, _to_bits(bits)))
    m = len(template)

    if not template or any(ch not in '01' for ch in template):
        raise ValueError("template must be a non-empty bit string.")

    block_count = len(data) // block_size

    if block_count == 0:
        raise ValueError("Input is too short for the selected block_size.")

    mean = (block_size - m + 1) / (2 ** m)
    variance = block_size * (1 / (2 ** m) - (2 * m - 1) / (2 ** (2 * m)))
    chi_square = 0

    for i in range(block_count):
        block = data[i * block_size:(i + 1) * block_size]
        pos = 0
        count = 0
        while pos <= block_size - m:
            if block[pos:pos + m] == template:
                count += 1
                pos += m
            else:
                pos += 1
        chi_square += (count - mean) ** 2 / variance

    return _igamc(block_count / 2, chi_square / 2)


def overlapping_template_test(bits: input_types, template: str = "111111111", block_size: int = 1032) -> float:

    '''Counts overlapping occurrences of a template and compares them with the NIST reference distribution for the overlapping template test. '''

    data = ''.join(map(str, _to_bits(bits)))
    m = len(template)
    block_count = len(data) // block_size

    if block_count == 0:
        raise ValueError("Input is too short for the selected block_size.")

    if any(ch not in "01" for ch in template): raise ValueError("template must contain only 0 and 1.")

    if m != 9 or block_size != 1032:
        raise ValueError("This implementation uses NIST probabilities only for template length 9 and block_size 1032.")

    probabilities = [0.364091, 0.185659, 0.139381, 0.100571, 0.070432, 0.139865]

    observed = [0] * 6

    for i in range(block_count):
        block = data[i * block_size:(i + 1) * block_size]

        count = sum(1 for j in range(block_size - m + 1) if block[j:j + m] == template)

        observed[min(count, 5)] += 1

    chi_square = sum((observed[i] - block_count * probabilities[i]) ** 2 / (block_count * probabilities[i]) for i in range(6))

    return _igamc(5.0 / 2.0, chi_square / 2.0)


def universal_statistical_test(bits: input_types, pattern_length: int = 7) -> float:

    '''Measures compressibility by tracking distances between repeated bit patterns.'''

    data = ''.join(map(str, _to_bits(bits)))
    n = len(data)
    l_value = pattern_length
    q_value = 10 * (2 ** l_value)
    k_value = n // l_value - q_value

    if k_value <= 0:
        raise ValueError("Input is too short for the selected pattern_length.")

    expected = {6: 5.2177052, 7: 6.1962507, 8: 7.1836656, 9: 8.1764248, 10: 9.1723243}.get(l_value)
    variance = {6: 2.954, 7: 3.125, 8: 3.238, 9: 3.311, 10: 3.356}.get(l_value)

    if expected is None or variance is None:
        raise ValueError("pattern_length must be between 6 and 10 in this implementation.")

    table: dict[str, int] = {}
    for i in range(q_value):
        pattern = data[i * l_value:(i + 1) * l_value]
        table[pattern] = i + 1

    total = 0
    for i in range(q_value, q_value + k_value):
        pattern = data[i * l_value:(i + 1) * l_value]
        distance = i + 1 - table.get(pattern, 0)
        table[pattern] = i + 1
        if distance > 0:
            total += log(distance, 2)

    fn = total / k_value
    sigma = sqrt(variance / k_value)

    return erfc(abs(fn - expected) / (sqrt(2) * sigma))


def linear_complexity_test(bits: input_types, block_size: int = 500) -> float:

    '''Estimates the LFSR complexity of each block using the Berlekamp-Massey algorithm.'''

    data = _to_bits(bits)
    block_count = len(data) // block_size

    if block_count == 0:
        raise ValueError("Input is too short for the selected block_size.")

    def berlekamp_massey(block: list[int]) -> int:
        c = [0] * block_size
        b = [0] * block_size
        c[0] = b[0] = 1
        l_complexity = 0
        m = -1
        for n_index in range(block_size):
            discrepancy = block[n_index]
            for i in range(1, l_complexity + 1):
                discrepancy ^= c[i] & block[n_index - i]
            if discrepancy == 1:
                temp = c[:]
                for j in range(block_size - n_index + m):
                    c[n_index - m + j] ^= b[j]
                if l_complexity <= n_index / 2:
                    l_complexity = n_index + 1 - l_complexity
                    m = n_index
                    b = temp
        return l_complexity

    mean = block_size / 2 + (9.0 + (-1) ** (block_size + 1)) / 36.0 - (block_size / 3.0 + 2 / 9.0) / (2 ** block_size)
    bins = [0] * 7

    for i in range(block_count):
        block = data[i * block_size:(i + 1) * block_size]
        complexity = berlekamp_massey(block)
        t_value = ((-1) ** block_size) * (complexity - mean) + 2 / 9.0
        index = 0 if t_value <= -2.5 else 1 if t_value <= -1.5 else 2 if t_value <= -0.5 else 3 if t_value <= 0.5 else 4 if t_value <= 1.5 else 5 if t_value <= 2.5 else 6
        bins[index] += 1

    probabilities = [0.01047, 0.03125, 0.12500, 0.50000, 0.25000, 0.06250, 0.020833]
    chi_square = sum((bins[i] - block_count * probabilities[i]) ** 2 / (block_count * probabilities[i]) for i in range(7))

    return _igamc(6.0 / 2, chi_square / 2)


def serial_test(bits: input_types, pattern_length: int = 3) -> dict[str, float]:

    '''Compares frequencies of all overlapping patterns of length m, m-1, and m-2.'''

    data = ''.join(map(str, _to_bits(bits)))
    n = len(data)

    if pattern_length < 2:
        raise ValueError("pattern_length must be at least 2.")
    extended = data + data[:pattern_length - 1]

    def psi(m: int) -> float:
        counts = Counter(extended[i:i + m] for i in range(n))
        return (2 ** m / n) * sum(v * v for v in counts.values()) - n

    psi_m = psi(pattern_length)
    psi_m1 = psi(pattern_length - 1)
    psi_m2 = psi(pattern_length - 2)
    delta1 = psi_m - psi_m1
    delta2 = psi_m - 2 * psi_m1 + psi_m2

    return {"p_value_1": _igamc(2 ** (pattern_length - 2), delta1 / 2), "p_value_2": _igamc(2 ** (pattern_length - 3), delta2 / 2)}


def approximate_entropy_test(bits: input_types, pattern_length: int = 3) -> float:
    
    '''Checks whether frequencies of adjacent pattern lengths have expected entropy.'''
    
    data = ''.join(map(str, _to_bits(bits)))
    n = len(data)
    
    if pattern_length < 1:
        raise ValueError("pattern_length must be positive.")

    def phi(m: int) -> float:
        extended = data + data[:m - 1]
        counts = Counter(extended[i:i + m] for i in range(n))
        return sum((count / n) * log(count / n) for count in counts.values() if count > 0)

    ap_en = phi(pattern_length) - phi(pattern_length + 1)
    chi_square = 2 * n * (log(2) - ap_en)
    return _igamc(2 ** (pattern_length - 1), chi_square / 2)


def cumulative_sums_test(bits: input_types, mode: str = "forward") -> float:
    
    '''Checks whether the cumulative random walk deviates too far from zero.'''
    
    data = _to_bits(bits)
    
    if mode not in ("forward", "backward"):
        raise ValueError("mode must be 'forward' or 'backward'.")
    x = [1 if bit else -1 for bit in data]
    
    if mode == "backward":
        x.reverse()
    cumulative = []
    current = 0
    
    for value in x:
        current += value
        cumulative.append(current)
        
    z_value = max(abs(v) for v in cumulative)
    n = len(data)
    if z_value == 0:
        return 1

    start = floor((-n / z_value + 1) / 4)
    end = floor((n / z_value - 1) / 4)
    total_1 = sum(norm.cdf((4 * k + 1) * z_value / sqrt(n)) - norm.cdf((4 * k - 1) * z_value / sqrt(n)) for k in range(start, end + 1))
    start = floor((-n / z_value - 3.0) / 4)
    end = floor((n / z_value - 1) / 4)
    total_2 = sum(norm.cdf((4 * k + 3) * z_value / sqrt(n)) - norm.cdf((4 * k + 1) * z_value / sqrt(n)) for k in range(start, end + 1))
    return 1 - total_1 + total_2


def random_excursions_test(bits: input_types) -> dict[int, float]:

    '''Counts visits to states -4..-1 and 1..4 within zero-delimited random-walk cycles.'''

    data = _to_bits(bits)
    x = [1 if bit else -1 for bit in data]
    walk = [0]
    for value in x:
        walk.append(walk[-1] + value)
    walk.append(0)

    zero_positions = [i for i, value in enumerate(walk) if value == 0]
    cycles = [walk[zero_positions[i]:zero_positions[i + 1] + 1] for i in range(len(zero_positions) - 1)]
    j = len(cycles)

    if j == 0:
        raise ValueError("No random-walk cycles were found.")

    probabilities = {1: [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.03125], 2: [0.75, 0.0625, 0.046875, 0.03515625, 0.0263671875, 0.0791015625],
        3: [0.8333333333, 0.0277777778, 0.0231481481, 0.0192901235, 0.0160751029, 0.0803755144], 4: [0.875, 0.015625, 0.013671875, 0.0119628906, 0.0104675293, 0.0732727051]}
    result: dict[int, float] = {}

    for state in [-4, -3, -2, -1, 1, 2, 3, 4]:
        abs_state = abs(state)
        observed = [0] * 6
        for cycle in cycles:
            visits = cycle.count(state)
            observed[min(visits, 5)] += 1
        chi_square = sum((observed[k] - j * probabilities[abs_state][k]) ** 2 / (j * probabilities[abs_state][k]) for k in range(6))
        result[state] = _igamc(5.0 / 2, chi_square / 2)

    return result


def random_excursions_variant_test(bits: input_types) -> dict[int, float]:
    
    '''Checks total visits to states -9..-1 and 1..9 in a cumulative random walk.'''
    
    data = _to_bits(bits)
    x = [1 if bit else -1 for bit in data]
    walk = [0]
    
    for value in x:
        walk.append(walk[-1] + value)
    j = walk.count(0) - 1
    
    if j <= 0:
        raise ValueError("No completed random-walk cycles were found.")

    result = {}
    
    for state in list(range(-9, 0)) + list(range(1, 10)):
        count = walk.count(state)
        result[state] = erfc(abs(count - j) / sqrt(2 * j * (4 * abs(state) - 2)))
    return result
