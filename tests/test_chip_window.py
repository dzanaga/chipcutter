from chipcutter import ChipWindow  # Import the ChipWindow class
import pytest

# Test the constructor


def test_constructor():
    window = ChipWindow(1, 5, 2, 6)
    assert window.start_row == 1
    assert window.end_row == 5
    assert window.start_col == 2
    assert window.end_col == 6

# Test updating attributes


def test_attributes():
    window = ChipWindow(1, 5, 2, 6)
    assert window.start_row == 1
    assert window.end_row == 5
    assert window.start_col == 2
    assert window.end_col == 6

    window.start_row = 10
    window.end_row = 20
    window.start_col = 30
    window.end_col = 40

    assert window.start_row == 10
    assert window.end_row == 20
    assert window.start_col == 30
    assert window.end_col == 40

# Test invalid values


def test_invalid_values():
    with pytest.raises(ValueError):
        ChipWindow(5, 1, 2, 6)  # Invalid row values

    with pytest.raises(ValueError):
        ChipWindow(1, 5, 6, 2)  # Invalid col values
