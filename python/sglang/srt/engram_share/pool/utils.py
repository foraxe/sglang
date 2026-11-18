# SPDX-License-Identifier: Apache-2.0
"""Small integer helpers for the peer pool."""


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def round_up(a: int, b: int) -> int:
    return cdiv(a, b) * b
