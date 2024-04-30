def ceiling_division(dividend: int, divisor: int) -> int:
    """Returns the ceiling of dividend / divisor.

    Args:
        dividend (int): The dividend.
        divisor (int): The divisor.

    Returns:
        int: The ceiling of the division result.

    Examples:
        >>> ceiling_division(5, 3)
        2
    """
    return -(dividend // -divisor)
