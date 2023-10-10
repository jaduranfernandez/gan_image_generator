from typing import List
import math
import numpy as np


def create_binary_list_from_int(number: int) -> List[int]:
    if number < 0 or type(number) is not int:
        raise ValueError("Only Positive integers are allowed")

    return [int(x) for x in list(bin(number))[2:]]


def create_data(max_value: int, n_samples: int):

    input_length = int(math.log(max_value, 2)) + 1
    sampled_integers = np.random.randint(0, int(max_value / 2), n_samples)

    # Generate a list of binary numbers for training.
    data = [create_binary_list_from_int(int(x * 2)) for x in sampled_integers]
    data = [([0] * (input_length - len(x))) + x for x in data]

    return np.array(data)