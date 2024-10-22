"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, List, Optional

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    """Multiplies two float numbers.

    Args:
    ----
        x: A float number.
        y: A float number.

    Returns:
    -------
        The product of x and y.

    """
    return x * y


def id(x: float) -> float:
    """Returns the input number.

    Args:
    ----
        x: A float number.

    Returns:
    -------
        The input number x.

    """
    return x


def add(x: float, y: float) -> float:
    """Adds two float numbers.

    Args:
    ----
        x: A float number.
        y: A float number.

    Returns:
    -------
        The sum of x and y.

    """
    return x + y


def neg(x: float) -> float:
    """Negates a float number.

    Args:
    ----
        x: A float number.

    Returns:
    -------
        The negation of x.

    """
    return -x


def lt(x: float, y: float) -> float:
    """Compares two float numbers.

    Args:
    ----
        x: A float number.
        y: A float number.

    Returns:
    -------
        True if x is less than y, False otherwise.

    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Compares two float numbers for equality.

    Args:
    ----
        x: A float number.
        y: A float number.

    Returns:
    -------
        True if x is equal to y, False otherwise.

    """
    return 1.0 if is_close(x, y) else 0.0


def max(x: float, y: float) -> float:
    """Returns the maximum of two float numbers.

    Args:
    ----
        x: A float number.
        y: A float number.

    Returns:
    -------
        The maximum of x and y.

    """
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Checks if two float numbers are close.

    Args:
    ----
        x: A float number.
        y: A float number.

    Returns:
    -------
        True if the absolute difference between x and y is less than 1e-2, False otherwise.

    """
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Calculates the sigmoid of a float number.

    Args:
    ----
        x: A float number.

    Returns:
    -------
        The sigmoid of x.

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Calculates the ReLU of a float number.

    Args:
    ----
        x: A float number.

    Returns:
    -------
        x if x is non-negative, 0 otherwise.

    """
    return x if x > 0 else 0.0


def log(x: float) -> float:
    """Calculates the natural logarithm of a float number.

    Args:
    ----
        x: A positive float number.

    Returns:
    -------
        The natural logarithm of x.

    """
    return math.log(x)


def exp(x: float) -> float:
    """Calculates the exponential of a float number.

    Args:
    ----
        x: A float number.

    Returns:
    -------
        The exponential of x.

    """
    return float(math.exp(x))


def log_back(x: float, back: float) -> float:
    """Computes the derivative of log times a second arg

    Args:
    ----
        x: A positive float number.
        back: A float number.

    Returns:
    -------
        The derivative of log(x) * back with respect to x.

    """
    return back / x


def inv(x: float) -> float:
    """Calculates the inverse of a float number.

    Args:
    ----
        x: A non-zero float number.

    Returns:
    -------
        The inverse of x.

    """
    return 1 / x


def inv_back(x: float, back: float) -> float:
    """Computes the derivative of inv times a second arg

    Args:
    ----
        x: A non-zero float number.
        back: A float number.

    Returns:
    -------
        The derivative of 1/x * back with respect to x.

    """
    return -back / x**2


def relu_back(x: float, back: float) -> float:
    """Computes the derivative of ReLU times a second arg

    Args:
    ----
        x: Input value
        back: Gradient from the next layer

    Returns:
    -------
        Gradient with respect to the input

    """
    return back * (x > 0)


# ## Task 0.3


def map(f: Callable[[float], float], lst: List[float]) -> List[float]:
    """Applies a function to each element of a list.

    Args:
    ----
        f: A function that takes a float and returns a float.
        lst: A list of float numbers.

    Returns:
    -------
        A new list with each element being the result of applying f to the corresponding element in lst.

    """
    return [f(x) for x in lst]


def zipWith(
    f: Callable[[float, float], float], lst1: List[float], lst2: List[float]
) -> List[float]:
    """Applies a function to corresponding elements of two lists.

    Args:
    ----
        f: A function that takes two float numbers and returns a float.
        lst1: The first list of float numbers.
        lst2: The second list of float numbers.

    Returns:
    -------
        A new list with each element being the result of applying f to the corresponding elements in lst1 and lst2.

    """
    return [f(x, y) for x, y in zip(lst1, lst2)]


def reduce(
    f: Callable[[float, float], float],
    lst: List[float],
    initial: Optional[float] = None,
) -> float:
    """Reduces a list to a single value by applying a function cumulatively.

    Args:
    ----
        f: A function that takes two float numbers and returns a float.
        lst: The list of float numbers to reduce.
        initial: Optional initial value for the reduction. If not provided, the first element of the list is used.

    Returns:
    -------
        The final reduced value.

    """
    if not lst and initial is None:
        raise ValueError("Cannot reduce an empty list without an initial value")

    result = initial if initial is not None else lst[0]
    start_index = 0 if initial is not None else 1

    for x in lst[start_index:]:
        result = f(result, x)
    return result


def negList(lst: List[float]) -> List[float]:
    """Negates all elements in a list of floats.

    Args:
    ----
        lst: A list of float numbers.

    Returns:
    -------
        A new list with each element being the negation of the corresponding element in lst.

    """
    return map(neg, lst)


def addLists(lst1: List[float], lst2: List[float]) -> List[float]:
    """Adds corresponding elements of two lists.

    Args:
    ----
        lst1: The first list of float numbers.
        lst2: The second list of float numbers.

    Returns:
    -------
        A new list with each element being the sum of the corresponding elements in lst1 and lst2.

    """
    return zipWith(add, lst1, lst2)


def sum(lst: List[float]) -> float:
    """Sums all elements in a list.

    Args:
    ----
        lst: The list of float numbers to sum.

    Returns:
    -------
        The sum of all elements in lst.

    """
    return reduce(add, lst, 0)


def prod(lst: List[float]) -> float:
    """Multiplies all elements in a list.

    Args:
    ----
        lst: The list of float numbers to multiply.

    Returns:
    -------
        The product of all elements in lst.

    """
    return reduce(mul, lst, 1)
