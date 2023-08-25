from typing import Tuple, Callable
from dataclasses import dataclass
import numpy as np

from chipcutter._version import __version__


__all__ = ['__version__']


@dataclass
class ChipWindow:

    start_row: int
    end_row: int
    start_col: int
    end_col: int

    def __post_init__(self):
        if self.start_row >= self.end_row:
            raise ValueError("start_row cannot be >= than end_row")

        if self.start_col >= self.end_col:
            raise ValueError("start_col cannot be >= than end_col")


class ChipCutter:

    def __init__(self,
                 arr: np.ndarray,
                 chip_shape: Tuple[int, int],
                 overlap: int = 0):
        """Provides methods to split a 3d numpy array (band, y, x) into a
        a batch of chips (chip, band, chip_shape[0], chip_shape[1]).
        Chips can overlap along the x and y dimensions. The `overlap` parameter
        determines how many pixels overlap from each side with neighboring
        chips.

        `split` method splits the image and `merge` method will compose the
        image back together. It assumes that x and y will not change after
        inference.

        Args:
            arr (np.ndarray): 3d numpy array (band, y, x)
            chip_shape (Tuple[int, int]): y, x shape of the chips
            overlap (int, optional): pixels overlap from each side with
            neighboring chips. Defaults to 0.

        Raises:
            ValueError: Overlap needs to be less than 2 times the minumum
            chip shape
        """

        self._chip_rows, self._chip_cols = chip_shape
        self._nbands, self._rows, self._cols = arr.shape
        self._overlap = overlap

        if (overlap * 2) >= min(chip_shape):
            raise ValueError(
                'overlap value should be less than min(chip_shape) / 2')

        self.windows = self._get_windows()
        self._arr = arr

    @staticmethod
    def _nchips(image_side: int, chip_side: int, overlap: int):
        """Compute number of chips along a dimension.
        If the number of pixels on the image dimension is not a multiple of the
        chip dimension, extra is a bool indicating
        if there is an extra chip at the end which overlaps
        further with the previous one."""

        inner_chip_side = chip_side - overlap
        inner_image_side = image_side - overlap

        nchips = inner_image_side // inner_chip_side
        extra = (inner_image_side % inner_chip_side) > 0

        return nchips, extra

    def _get_windows(self):

        chip_rows, chip_cols = self._chip_rows, self._chip_cols
        rows, cols = self._rows, self._cols
        overlap = self._overlap

        r_chips, r_extra = self._nchips(rows, chip_rows, overlap)
        c_chips, c_extra = self._nchips(cols, chip_cols, overlap)

        windows = []
        r_start = 0
        c_start = 0

        r_stride = chip_rows - overlap
        c_stride = chip_cols - overlap

        for _ in range(r_chips):
            for _ in range(c_chips):
                w = ChipWindow(r_start, r_start + chip_rows,
                               c_start, c_start + chip_cols)
                windows.append(w)

                c_start += c_stride
            c_start = 0
            r_start += r_stride

        # extra row chips
        if r_extra:
            for c in range(c_chips):
                w = ChipWindow(rows - chip_rows, rows,
                               c * c_stride, c * c_stride + chip_cols)
                windows.append(w)

        # extra col chips
        if c_extra:
            for r in range(r_chips):
                w = ChipWindow(r * r_stride, r * r_stride + chip_rows,
                               cols - chip_cols, cols)
                windows.append(w)

        # bottom right corner block
        if r_extra and r_extra:
            w = ChipWindow(rows - chip_rows, rows,
                           cols - chip_cols, cols)
            windows.append(w)

        return windows

    def split(self):
        batch_size = len(self.windows)
        batch = np.zeros((batch_size,
                          self._nbands,
                          self._chip_rows,
                          self._chip_cols),
                         dtype=self._arr.dtype)

        for i, w in enumerate(self.windows):
            batch[i, ...] = self._arr[:,
                                      w.start_row:w.end_row,
                                      w.start_col:w.end_col]
        return batch

    def _merge_band(self, pred_batch):
        pred = np.zeros((self._rows, self._cols), dtype=np.float32)
        for i, w in enumerate(self.windows):
            start_row, end_row = self._pred_start_end_rows(w)
            start_col, end_col = self._pred_start_end_cols(w)

            block_start_row = start_row - w.start_row
            block_end_row = end_row + self._chip_rows - w.end_row

            block_start_col = start_col - w.start_col
            block_end_col = end_col + self._chip_cols - w.end_col

            pred[start_row:end_row,
                 start_col:end_col] = pred_batch[i,
                                                 block_start_row:block_end_row,
                                                 block_start_col:block_end_col]

        return pred

    def merge(self, pred_batch) -> np.ndarray:
        """Merges batch back into image.

        If the batch has 3 dimensions (batch, y, x), it will return a
        2d image (y, x).
        If the batch has 4 dimensions, it assumes that the batch dimension is
        the first axis (batch, band, y, x) it will return a
        3d image (band, y, x).

        Args:
            pred_batch (np.ndarray): batch to merge back

        Returns:
            np.ndarray: Merged image
        """
        if pred_batch.ndim == 3:
            return self._merge_band(pred_batch)
        elif pred_batch.ndim == 4:
            return np.array([self._merge_band(pred_batch[:, i, ...])
                             for i in range(pred_batch.shape[1])])

    def _pred_start_end_rows(self, w):
        """Get start, end rows when merging predictions. If there is an overlap
        only chips on the rows at the start and end extend beyond overlap area
        for merging."""
        overlap = self._overlap
        left_half_overlap = int(np.floor(overlap / 2))
        right_half_overlap = int(np.ceil(overlap / 2))
        if w.start_row == 0:
            start_row = w.start_row
        else:
            start_row = w.start_row + right_half_overlap

        if w.end_row == self._rows:
            end_row = w.end_row
        else:
            end_row = w.end_row - left_half_overlap

        return start_row, end_row

    def _pred_start_end_cols(self, w):
        """Get start, end cols when merging predictions. If there is an overlap
        only chips on the cols at the start and end extend beyond overlap area
        for merging."""
        overlap = self._overlap
        left_half_overlap = int(np.floor(overlap / 2))
        right_half_overlap = int(np.ceil(overlap / 2))
        if w.start_col == 0:
            start_col = w.start_col
        else:
            start_col = w.start_col + right_half_overlap

        if w.end_col == self._cols:
            end_col = w.end_col
        else:
            end_col = w.end_col - left_half_overlap

        return start_col, end_col

    def predict(self,
                inference_func: Callable,
                max_batch_size: int = 128,
                **kwargs) -> np.ndarray:
        """Compute inference. `inference_func` takes as input a batch and
        the optional **kwargs and computes the
        inference, returning a batch with at least 3 dimensions (batch, y, x)
        or 4 (batch, band, y, x)

        Args:
            inference_func (Callable): Computes batch inference, returns
            prediction/probabilities inference.
            max_batch_size (int, optional): The input array will be split in
            a certain number of chips, forming sometimes a very large batch.
            To avoid saturating the memory, the batch is split and inference
            is computed on the chunked batches. Defaults to 128.

        Returns:
            np.ndarray: Mosaicked prediction
        """

        mbs = max_batch_size

        batch = self.split()
        batch_size = batch.shape[0]

        pred_batch = None
        chunks = int(np.ceil(batch_size / mbs))

        for i in range(chunks):
            chunked_batch = batch[mbs * i:min(mbs * (i + 1), batch_size), ...]
            print(f'batch {i}: chunked_batch.shape: {chunked_batch.shape}')
            if pred_batch is None:
                pred_batch = inference_func(chunked_batch, **kwargs)
            else:
                pred_batch = np.concatenate([pred_batch,
                                             inference_func(chunked_batch)],
                                            axis=0)

        pred = self.merge(pred_batch)

        return pred
