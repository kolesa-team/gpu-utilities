from gpu_utilities.utils.bytes import UnitSize
import pytest

class TestUnitSize:
    def test_multiplier(self):
        assert 1 == UnitSize.B.multiplier()
        assert 2**10 == UnitSize.KiB.multiplier()
        assert 2**20 == UnitSize.MiB.multiplier()
        assert 2**30 == UnitSize.GiB.multiplier()
        assert 2**40 == UnitSize.TiB.multiplier()
        assert 2**50 == UnitSize.PiB.multiplier()

    def test_traditional_name(self):
        assert 'byte' == UnitSize.B.traditional_name()
        assert 'kibibyte' == UnitSize.KiB.traditional_name()
        assert 'mebibyte' == UnitSize.MiB.traditional_name()
        assert 'gibibyte' == UnitSize.GiB.traditional_name()
        assert 'tebibyte' == UnitSize.TiB.traditional_name()
        assert 'pebibyte' == UnitSize.PiB.traditional_name()


@pytest.mark.parametrize(
    "expected,bytes",
    [
        (0, 0),
        (1024, 1024),
        (5000,5000),
    ]
)
def test_B_from_bytes(expected: int, bytes: int):
    assert 0 == UnitSize.B.from_bytes(0)
