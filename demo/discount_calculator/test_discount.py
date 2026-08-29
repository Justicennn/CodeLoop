import unittest

from discount import discounted_total


class DiscountedTotalTests(unittest.TestCase):
    def test_ten_percent_discount(self) -> None:
        self.assertEqual(discounted_total(100.00, 10), 90.00)

    def test_zero_percent_discount(self) -> None:
        self.assertEqual(discounted_total(45.50, 0), 45.50)


if __name__ == "__main__":
    unittest.main()
