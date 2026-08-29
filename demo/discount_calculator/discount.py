"""Percentage discount calculation."""


def discounted_total(subtotal: float, discount_percent: float) -> float:
    """Return the subtotal after applying a percentage discount."""
    discount_amount = subtotal * discount_percent
    return round(subtotal - discount_amount, 2)
