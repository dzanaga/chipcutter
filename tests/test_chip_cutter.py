import numpy as np
import pytest
from chipcutter import ChipCutter

# Mock inference function for testing


def mock_inference(batch, **kwargs):
    return np.zeros((batch.shape[0], batch.shape[2], batch.shape[3]))

# Test ChipCutter class


def test_chip_splitter_constructor_valid():
    arr = np.random.rand(3, 256, 256)  # Example 3-band image
    chip_shape = (128, 128)
    overlap = 0
    splitter = ChipCutter(arr, chip_shape, overlap)
    assert splitter._chip_rows == chip_shape[0]
    assert splitter._chip_cols == chip_shape[1]
    assert splitter._nbands == arr.shape[0]
    assert splitter._rows == arr.shape[1]
    assert splitter._cols == arr.shape[2]
    assert splitter._overlap == overlap
    assert len(splitter.windows) > 0


def test_chip_splitter_constructor_invalid_overlap():
    arr = np.random.rand(3, 256, 256)  # Example 3-band image
    chip_shape = (128, 128)
    overlap = 65  # More than half of the chip shape
    with pytest.raises(ValueError):
        splitter = ChipCutter(arr, chip_shape, overlap)


def test_chip_splitter_split_merge_nooverlap():
    arr = np.random.rand(3, 256, 256)  # Example 3-band image
    chip_shape = (128, 128)
    overlap = 0
    splitter = ChipCutter(arr, chip_shape, overlap)
    batch = splitter.split()
    assert batch.shape[0] == len(splitter.windows)
    assert batch.shape[1] == arr.shape[0]
    assert batch.shape[2] == chip_shape[0]
    assert batch.shape[3] == chip_shape[1]

    merged = splitter.merge(batch)
    print(np.isclose(arr, merged).mean())
    assert np.allclose(arr, merged)


def test_chip_splitter_split_merge_overlap():
    arr = np.random.rand(3, 256, 256)  # Example 3-band image
    chip_shape = (128, 128)
    overlap = 16
    splitter = ChipCutter(arr, chip_shape, overlap)
    batch = splitter.split()
    assert batch.shape[0] == len(splitter.windows)
    assert batch.shape[1] == arr.shape[0]
    assert batch.shape[2] == chip_shape[0]
    assert batch.shape[3] == chip_shape[1]

    merged = splitter.merge(batch)
    assert np.allclose(arr, merged)

    merged = splitter.merge(batch[:, 0, :, :])
    assert np.allclose(arr[0], merged)


def test_chip_splitter_predict():
    arr = np.random.rand(3, 256, 256)  # Example 3-band image
    chip_shape = (128, 128)
    overlap = 16
    splitter = ChipCutter(arr, chip_shape, overlap)

    pred = splitter.predict(mock_inference)
    assert pred.shape == (arr.shape[1], arr.shape[2])  # 2D merged prediction


# Run the tests
if __name__ == "__main__":
    pytest.main()
