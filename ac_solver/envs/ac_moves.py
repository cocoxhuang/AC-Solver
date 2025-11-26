from ac_solver.envs.utils import simplify_presentation, simplify_relator
import numpy as np

def concatenate_relators(presentation, max_relator_length, i, j, sign, lengths):
    """
    Given a presentation <r_0, r_1>, returns a new presentation where r_i is replaced by r_i r_j^{sign}.

    Parameters:
    presentation: A Numpy Array representing a presentation
    max_relator_length: An int. Maximum length the concatenated relator is allowed to have.
                        If length of the concatenated relator (after simplification) is greater than this integer,
                        the original presentation is returned without any changes.
                        The only simplifications applied are free reductions and not cyclical reductions as the latter
                        correspond to conjugations on a given word.
    i: 0 or 1, index of the relator to change.
    j: 0 or 1, but not equal to i.
    sign: +1 or -1, whether to invert r_j before concatenation.
    lengths: A list of lengths of words in presentation.

    Returns:
    (resultant_presentation, lengths_of_resultant_presentations)
    resultant_presentation is the presentation with r_i possibly replaced with r_i r_j^{sign}.
    lengths_of_resultant_presentations is the list of lengths of words in the resultant presentation.
    """
    assert all(
        [
            i in [0, 1],
            j in [0, 1],
            i == 1 - j,
        ]
    ), f"expect i and j to be 0 or 1 and i != j; got i = {i}, j = {j}"

    assert sign in [1, -1], f"expect sign to be +1 or -1, received {sign}"

    # get r_i
    presentation = presentation.copy()
    relator1 = presentation[i * max_relator_length : (i + 1) * max_relator_length]

    # get r_j or r_j^{-1} depending on sign
    # TODO: really need to understand this
    if sign == 1:
        relator2 = presentation[j * max_relator_length : (j + 1) * max_relator_length]
    elif j:
        relator2 = -presentation[
            (j + 1) * max_relator_length - 1 : j * max_relator_length - 1 : -1
        ]
    else:
        relator2 = -presentation[max_relator_length - 1 :: -1]

    relator1_nonzero = relator1[relator1 != 0]
    relator2_nonzero = relator2[relator2 != 0]

    len1 = len(relator1_nonzero)
    len2 = len(relator2_nonzero)

    acc = 0
    while (
        acc < min(len1, len2) and relator1_nonzero[-1 - acc] == -relator2_nonzero[acc]
    ):
        acc += 1

    new_size = len1 + len2 - 2 * acc

    if new_size <= max_relator_length:
        lengths[i] = new_size
        presentation[i * max_relator_length : i * max_relator_length + len1 - acc] = (
            relator1_nonzero[: len1 - acc]
        )
        presentation[
            i * max_relator_length + len1 - acc : i * max_relator_length + new_size
        ] = relator2_nonzero[acc:]
        presentation[
            i * max_relator_length + new_size : (i + 1) * max_relator_length
        ] = 0

    return presentation, lengths


def conjugate(presentation, max_relator_length, i, j, sign, lengths):
    """
    Given a presentation <r_0, r_1>, returns a new presentation where r_i is replaced by x_j^{sign} r_i x_j^{-sign}.

    Parameters:
    presentation: A Numpy Array representing a presentation
    max_relator_length: An int. Maximum length the concatenated relator is allowed to have.
                        If length of the concatenated relator (after simplification) is greater than this integer,
                        the original presentation is returned without any changes.
                        The only simplifications applied are free reductions and not cyclical reductions as the latter
                        correspond to conjugations on a given word.
    i: 0 or 1, index of the relator to change.
    j: 1 or 2, index of the generator to conjugate with.
    sign: +1 or -1, whether to invert x_j before concatenation.
    lengths: A list of lengths of words in presentation.

    Returns:
    (resultant_presentation, lengths_of_resultant_presentations)
    resultant_presentation is the presentation with r_i possibly replaced with x_j^{sign} r_i x_j^{-sign}.
    lengths_of_resultant_presentations is the list of lengths of words in the resultant presentation.

    """
    # TODO: perhaps i and j should be more uniformly both in [0, 1].
    assert all(
        [i in [0, 1], j in [1, 2]]
    ), f"expect i to be 0 and 1 and j to be 1 or 2; got i = {i}, j = {j}"

    assert sign in [1, -1], f"expect sign to be +1 or -1, received {sign}"

    presentation = presentation.copy()
    relator = presentation[i * max_relator_length : (i + 1) * max_relator_length]
    relator_nonzero = relator[relator.nonzero()]
    relator_size = len(relator_nonzero)

    # get the generator that is to be appended on the left
    generator = sign * j

    # TODO: again here, it will be good to use simplify_relator

    # check whether we will need to cancel any generators at the beginning and at the end
    start_cancel = 1 if relator_nonzero[0] == -generator else 0
    end_cancel = 1 if relator_nonzero[-1] == generator else 0

    # get the size of the resultant relator after cancellation
    new_size = relator_size + 2 - 2 * (start_cancel + end_cancel)

    # update lengths and presentation
    if new_size <= max_relator_length:
        lengths = lengths.copy()
        lengths[i] = new_size

        presentation[
            i * max_relator_length
            + 1
            - start_cancel : i * max_relator_length
            + 1
            + relator_size
            - 2 * start_cancel
            - end_cancel
        ] = relator_nonzero[start_cancel : relator_size - end_cancel]

        if not start_cancel:
            presentation[i * max_relator_length] = generator

        if not end_cancel:
            presentation[
                i * max_relator_length + relator_size + 1 - 2 * start_cancel
            ] = -generator

        if start_cancel and end_cancel:
            presentation[
                i * max_relator_length
                + new_size : i * max_relator_length
                + new_size
                + 2
            ] = 0

    return presentation, lengths

def substituition(r1: np.ndarray, r2: np.ndarray, action: tuple) -> tuple:
    """
    Reference: https://github.com/Math-AI-Caltech/AC-SolverX/blob/main/classical_search/AC_substitution_optimized.ipynb
    Applies a substitution to relators r1 and r2 based on the given action.

    Args:
        r1 (np.ndarray): The first relator, with the padded zeros removed.
        r2 (np.ndarray): The second relator, with the padded zeros removed.
        action (tuple): A tuple (relator, i, j, inverse_indicator) where
            relator: 0 or 1, index of the relator to change.
            i: int, rotation amount for r1.
            j: int, rotation amount for r2.
            inverse_indicator: boolean, whether to invert r2 before substitution.
            Note that r2 is always the lexicographically larger relator.
    """
    relator, i, j, inverse_indicator = action
    r2 = invert(r2) if inverse_indicator else r2

    r1 = np.roll(r1, i)
    r2 = np.roll(r2, j)
    neighbour = np.concatenate([r1, r2])

    if relator == 0:    # then replace r1
        return neighbour, r2
    else:               # then replace r2
        return r1, neighbour
    
def lex_cmp(c1 : int, c2: int) -> bool:
    """
    Compare two charaters c1 and c2 in lexiographic order,
    where the order is defined as: 
    2 < -2 < -1 < 1, i.e.
    Y < y < X < x
    Returns True if c1 >= c2.
    """
    assert c1 != c2, "Cannot compare equal characters."
    if c1 == 2:
        return False
    elif c1 == -2:
        return True if c2 == 2 else False
    elif c1 == -1:
        return False if c2 == 1 else True
    else:   # c1 == 1 or 'x'
        return True
    
def lex_cmp_array(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Compare two numpy arrays of shape (n, 2) with bool types in lexiographic order,
    where the order is defined as:
    2 < -2 < -1 < 1, i.e.
    Y < y < X < x.
    Returns True if a > b.
    """
    assert a.shape == b.shape, "Cannot compare arrays of different shapes."
    for x, y in zip(a, b):
        if x == y:
            continue
        else:
            return lex_cmp(x, y)
    
    # If we reach here the arrays are equal.
    return False

def find_minimal_rotation(rel: np.ndarray) -> np.ndarray:
    '''
    Find the minimal rotation of a relator using Booth's algorithm.

    Uses the ordering `2 < -2 < -1 < 1` for the characters, 
    i.e. `Y < y < X < x`.

    TODO: verify correctness.
    '''
    n = len(rel)
    rel = np.concatenate([rel, rel])
    f = np.full(2 * n, -1, dtype=np.int32)
    k = 0
    for j in range(1, 2 * n):
        i = f[j - k - 1]
        while i != -1 and (not rel[j] == rel[k + i + 1]):
            if not lex_cmp(rel[j], rel[k + i + 1]):
                k = j - i - 1
            i = f[i]
        if i == -1 and (not rel[j] == rel[k]):
            if not lex_cmp(rel[j], rel[k]):
                k = j
            f[j - k] = -1
        else:
            f[j - k] = i + 1
    return rel[k:k + n]

def invert(r: np.ndarray) -> np.ndarray:
    """Returns the inverse of a relator r."""
    return -r[::-1]

def canonical_relator(r: np.ndarray) -> np.ndarray:
    """
    Returns the canonical form of a relator r, which is the
    lexicographically smallest rotation of r or its inverse.
    """
    if len(r) == 0:
        return r
    r_min = find_minimal_rotation(r)
    inv_min = find_minimal_rotation(invert(r))
    if lex_cmp_array(r_min, inv_min):
        return inv_min
    return r_min
    
def canonical_pair(r1: np.ndarray, r2: np.ndarray, len_r1: int, len_r2: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Returns the canonical pair of relators (r1, r2) such that
    (0) r1 and r2 are converted to their canonical_relator forms, and
    (1) len(r1) is smaller than len(r2), or
    (2) if they are equal, r1 is cyclically smaller than r2.
    """
    r1 = canonical_relator(r1)
    r2 = canonical_relator(r2)
    if len(r1) > len(r2):
        return r2, r1, len_r2, len_r1
    elif (len(r1) == len(r2) and lex_cmp_array(r1, r2)):
        return r2, r1, len_r2, len_r1
    else:
        return r1, r2, len_r1, len_r2

def ACMove(action: tuple, presentation: np.ndarray, max_relator_length: int, 
           lengths: list, cylically_reduce=True) -> tuple[np.ndarray, list]:
    """
    Applies an AC move (concatenation or conjugation) to a presentation and returns the resultant presentation.
    The move to apply and the relator it is applied to are decided by move_id.

    Parameters:
    action: An tuple (relator, i, j, inverse_indicator) where:
        relator: 0 or 1, index of the relator to change.
        i: the rotation amount for r1.
        j: the rotation amount for r2.
        inverse_indicator: whether to invert r2 before substitution.
        Note that r2 is always the lexicographically larger relator.
    presentation: A NumPy Array representation the input presentation <r_0, r_1>.
        Each relator is represented as a fixed-length array of integers, padded with zeros on the right.
    max_relator_length: The maximum length a relator is allowed to take.
                        If the application of an AC move results in a relator with length larger than max_relator_length,
                        the original presentation is returned.
    lengths: A list of lengths of words in the presentation.
    cylically_reduce: A bool; whether to cyclically reduce words in the resultant presentation or not.
    """

    # get non-padded relators
    r1, r2 = presentation[:max_relator_length], presentation[max_relator_length:]
    r1, r2 = r1[r1.nonzero()], r2[r2.nonzero()]

    # apply the move
    r1, r2 = substituition(r1, r2, action)

    # simplify r1 and r2
    try:
        r1, len_r1 = simplify_relator(r1, max_relator_length, cylically_reduce=True, padded=False)
        r2, len_r2 = simplify_relator(r2, max_relator_length, cylically_reduce=True, padded=False)
    except Exception as e:
        r1, len_r1 = r1, len(r1)
        r2, len_r2, max_relator_length = r2, len(r2), max_relator_length
        print("r1:", r1)
        print("r2:", r2)
        print("len_r1:", len_r1)
        print("len_r2:", len_r2)
        print("max_relator_length:", max_relator_length)
        raise ValueError("Error simplifying r2")

    # change to canonical pair
    r1, r2, len_r1, len_r2 = canonical_pair(r1, r2, len_r1, len_r2)

    # padd back to fixed length
    r1 = np.concatenate([r1, np.zeros(max_relator_length - len(r1), dtype=int)])
    r2 = np.concatenate([r2, np.zeros(max_relator_length - len(r2), dtype=int)])
    presentation = np.concatenate([r1, r2])
    lengths = [len_r1, len_r2]

    # if move_id in range(0, 4):
    #     move_id += 1
    #     i = move_id % 2
    #     j = 1 - i
    #     sign_parity = ((move_id - i) // 2) % 2
    #     sign = (-1) ** sign_parity
    #     move = concatenate_relators
    # elif move_id in range(4, 12):
    #     move_id += 1
    #     i = move_id % 2
    #     jp = ((move_id - i) // 2) % 2  # = 0 or 1
    #     sign_parity = ((move_id - i - 2 * jp) // 4) % 2
    #     j = jp + 1  # = 1 or 2
    #     sign = (-1) ** sign_parity
    #     move = conjugate

    # presentation, lengths = move(
    #     presentation=presentation,
    #     max_relator_length=max_relator_length,
    #     i=i,
    #     j=j,
    #     sign=sign,
    #     lengths=lengths,
    # )

    # TODO: simplify_presentation seems to do something non-trivial even when
    # cyclical=False. I ran into trouble by putting an `if cyclical==False` cond
    # before the next lines of code.
    # This is confusing because I thought cojugate and concatenate_relators
    # already do the cyclical=False simplification.

    # presentation, lengths = simplify_presentation(
    #     presentation=presentation,
    #     max_relator_length=max_relator_length,
    #     lengths_of_words=lengths,
    #     cylically_reduce=cylically_reduce,
    # )

    return presentation, lengths
