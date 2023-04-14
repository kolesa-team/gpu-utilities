from gpu_utilities.utils.bytes import UnitSize, convert
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
        assert "byte" == UnitSize.B.traditional_name()
        assert "kibibyte" == UnitSize.KiB.traditional_name()
        assert "mebibyte" == UnitSize.MiB.traditional_name()
        assert "gibibyte" == UnitSize.GiB.traditional_name()
        assert "tebibyte" == UnitSize.TiB.traditional_name()
        assert "pebibyte" == UnitSize.PiB.traditional_name()


@pytest.mark.parametrize(
    "expected,bytes",
    [
        (0, 0),
        (1024, 1024),
        (5000, 5000),
    ],
)
def test_B_from_bytes(expected: int, bytes: int):
    assert expected == UnitSize.B.from_bytes(bytes)


@pytest.mark.parametrize(
    "expected,bytes",
    [
        (0, 0),
        (1024, 1024),
        (5000, 5000),
    ],
)
def test_B_to_bytes(expected: int, bytes: int):
    assert expected == UnitSize.B.to_bytes(bytes)


@pytest.mark.parametrize(
    "expected,bytes",
    [
        (0, 1000),
        (0, 1023),
        (1, 1024),
        (10, 10240),
    ],
)
def test_KiB_from_bytes(expected: int, bytes: int):
    assert expected == UnitSize.KiB.from_bytes(bytes)


@pytest.mark.parametrize(
    "expected,bytes",
    [
        (0, 0),
        (1024, 1),
        (2048, 2),
        (10240, 10),
    ],
)
def test_KiB_to_bytes(expected: int, bytes: int):
    assert expected == UnitSize.KiB.to_bytes(bytes)


@pytest.mark.parametrize(
    "expected,bytes",
    [
        (0, 100000),
        (0, 102300),
        (1, 1049000),
        (10, 10490000),
    ],
)
def test_MiB_from_bytes(expected: int, bytes: int):
    assert expected == UnitSize.MiB.from_bytes(bytes)


@pytest.mark.parametrize(
    "expected,bytes",
    [
        (0, 0),
        (1048576, 1),
        (10485760, 10),
    ],
)
def test_MiB_to_bytes(expected: int, bytes: int):
    assert expected == UnitSize.MiB.to_bytes(bytes)


@pytest.mark.parametrize(
    "expected,size,from_unit,target_unit",
    [
        (0, 0, UnitSize.B, UnitSize.KiB),
        (0, 0, UnitSize.KiB, UnitSize.KiB),
        (0, 0, UnitSize.MiB, UnitSize.KiB),
        (0, 0, UnitSize.GiB, UnitSize.KiB),
        (0, 0, UnitSize.TiB, UnitSize.KiB),
        (0, 0, UnitSize.PiB, UnitSize.KiB),
        (0, 0, UnitSize.KiB, UnitSize.MiB),
        (0, 0, UnitSize.TiB, UnitSize.GiB),
        (1, 1024, UnitSize.B, UnitSize.KiB),
        (1, 1024, UnitSize.KiB, UnitSize.MiB),
        (1, 1024, UnitSize.MiB, UnitSize.GiB),
        (1, 1024, UnitSize.GiB, UnitSize.TiB),
        (1, 1024, UnitSize.TiB, UnitSize.PiB),
        (1, 1024 * 1024, UnitSize.B, UnitSize.MiB),
        (1, 1024 * 1024 * 1024, UnitSize.B, UnitSize.GiB),
        (1024, 1, UnitSize.KiB, UnitSize.B),
        (1024 * 1024, 1, UnitSize.MiB, UnitSize.B),
        (1024 * 1024 * 1024, 1, UnitSize.GiB, UnitSize.B),
        (1, 1600, UnitSize.B, UnitSize.KiB),
    ],
)
def test_convert(
    expected, size: int, from_unit: UnitSize, target_unit: UnitSize
):
    assert expected == convert(size, from_unit, target_unit)
