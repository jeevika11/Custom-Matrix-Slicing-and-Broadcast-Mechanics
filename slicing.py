"""
slicing.py
----------------------------------------------------
Task 7: Custom Matrix Slicing

Implements multidimensional tensor slicing from scratch.

Important:
    Standard NumPy slicing operators such as:

        array[1:4, 2:5]

    are NOT used for the actual slicing operation.

The implementation calculates:
    - Output shape
    - Source coordinates
    - Source flat-buffer offsets
    - Destination positions

using explicit tensor strides.
"""

import numpy as np

from tensor_engine import TensorEngine


class CustomSlicer:
    """
    Performs tensor slicing using manually calculated
    offsets and strides.
    """

    def __init__(self, tensor):
        if isinstance(tensor, TensorEngine):
            self.tensor = tensor
        else:
            self.tensor = TensorEngine(tensor)

    # ========================================================
    # SLICE NORMALIZATION
    # ========================================================

    def _normalize_slice(
        self,
        start,
        stop,
        step,
        dimension
    ):
        """
        Normalize a Python-style slice for one dimension.

        Only the slice values are interpreted here.
        NumPy slicing is not used.
        """

        if step == 0:
            raise ValueError(
                "Slice step cannot be zero."
            )

        if step is None:
            step = 1

        if start is None:
            start = 0 if step > 0 else dimension - 1

        if stop is None:
            stop = dimension if step > 0 else -1

        if start < 0:
            start += dimension

        if stop < 0:
            stop += dimension

        if step > 0:

            start = max(
                0,
                min(start, dimension)
            )

            stop = max(
                0,
                min(stop, dimension)
            )

        else:

            start = max(
                -1,
                min(start, dimension - 1)
            )

            stop = max(
                -1,
                min(stop, dimension - 1)
            )

        return start, stop, step

    # ========================================================
    # RANGE GENERATION
    # ========================================================

    @staticmethod
    def _slice_indices(
        start,
        stop,
        step
    ):
        """
        Generate indices manually.

        This avoids NumPy slicing.
        """

        indices = []

        current = start

        if step > 0:

            while current < stop:
                indices.append(current)
                current += step

        else:

            while current > stop:
                indices.append(current)
                current += step

        return indices

    # ========================================================
    # CUSTOM SLICE
    # ========================================================

    def slice(
        self,
        slices
    ):
        """
        Perform a multidimensional slice.

        Example:

            tensor.shape = (4, 5)

            slices = (
                slice(1, 3),
                slice(2, 5)
            )

        The source values are obtained using manually
        calculated flat-buffer offsets.
        """

        if len(slices) != self.tensor.ndim:
            raise ValueError(
                f"Expected {self.tensor.ndim} "
                f"slice objects, got {len(slices)}."
            )

        index_lists = []

        for axis, slice_object in enumerate(slices):

            if not isinstance(
                slice_object,
                slice
            ):
                raise TypeError(
                    "Each dimension must use "
                    "a slice object."
                )

            start, stop, step = (
                self._normalize_slice(
                    slice_object.start,
                    slice_object.stop,
                    slice_object.step,
                    self.tensor.shape[axis]
                )
            )

            indices = self._slice_indices(
                start,
                stop,
                step
            )

            index_lists.append(indices)

        output_shape = tuple(
            len(indices)
            for indices in index_lists
        )

        total_elements = 1

        for dimension in output_shape:
            total_elements *= dimension

        result = np.empty(
            output_shape,
            dtype=self.tensor.dtype
        )

        # ----------------------------------------------------
        # Empty result
        # ----------------------------------------------------

        if total_elements == 0:
            return result

        # ----------------------------------------------------
        # Cartesian coordinate generation
        #
        # Implemented manually instead of using
        # NumPy multidimensional slicing.
        # ----------------------------------------------------

        coordinates = [0] * self.tensor.ndim

        self._fill_result(
            result=result,
            index_lists=index_lists,
            axis=0,
            coordinates=coordinates
        )

        return result

    # ========================================================
    # RECURSIVE RESULT FILL
    # ========================================================

    def _fill_result(
        self,
        result,
        index_lists,
        axis,
        coordinates
    ):
        """
        Recursively walk through output coordinates.

        Source offsets are calculated using the tensor's
        explicit strides.
        """

        if axis == self.tensor.ndim:

            source_offset = 0

            for coordinate, stride in zip(
                coordinates,
                self.tensor.strides
            ):
                source_offset += (
                    coordinate * stride
                )

            value = self.tensor.flat[
                source_offset
            ]

            self._set_result_value(
                result,
                coordinates,
                index_lists,
                value
            )

            return

        for output_index, source_index in enumerate(
            index_lists[axis]
        ):

            coordinates[axis] = source_index

            self._fill_result(
                result=result,
                index_lists=index_lists,
                axis=axis + 1,
                coordinates=coordinates
            )

    # ========================================================
    # RESULT INDEX CONVERSION
    # ========================================================

    @staticmethod
    def _set_result_value(
        result,
        source_coordinates,
        index_lists,
        value
    ):
        """
        Convert source coordinates into destination
        coordinates and assign the value.

        The assignment is performed through a flat
        representation of the result.
        """

        output_coordinates = []

        for axis, source_coordinate in enumerate(
            source_coordinates
        ):

            output_index = 0

            for index, value_index in enumerate(
                index_lists[axis]
            ):

                if value_index == source_coordinate:
                    output_index = index
                    break

            output_coordinates.append(
                output_index
            )

        # Convert destination coordinates into
        # a flat offset manually.

        if len(output_coordinates) == 0:
            result.flat[0] = value
            return

        offset = 0

        stride = 1

        for axis in range(
            len(output_coordinates) - 1,
            -1,
            -1
        ):

            offset += (
                output_coordinates[axis]
                * stride
            )

            stride *= result.shape[axis]

        result.flat[offset] = value


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def custom_slice(
    array,
    slices
):
    """
    Convenience wrapper for CustomSlicer.
    """

    slicer = CustomSlicer(array)

    return slicer.slice(slices)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CUSTOM SLICING TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Create test tensor
    # --------------------------------------------------------

    data = np.arange(
        24,
        dtype=np.float64
    ).reshape(4, 6)

    print("\nOriginal Matrix")
    print("-" * 60)

    print(data)

    # --------------------------------------------------------
    # Custom slice
    # --------------------------------------------------------

    slicer = CustomSlicer(data)

    result = slicer.slice(
        (
            slice(1, 3),
            slice(2, 6)
        )
    )

    print("\nCustom Slice Result")
    print("-" * 60)

    print(result)

    # --------------------------------------------------------
    # Expected result
    #
    # This is used ONLY for verification.
    # The actual slicing above is custom.
    # --------------------------------------------------------

    expected = np.array(
        [
            [8, 9, 10, 11],
            [14, 15, 16, 17]
        ],
        dtype=np.float64
    )

    print("\nExpected Result")
    print("-" * 60)

    print(expected)

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    if np.array_equal(
        result,
        expected
    ):
        print(
            "\nCustom slicing verification : PASS"
        )
    else:
        print(
            "\nCustom slicing verification : FAIL"
        )

    # --------------------------------------------------------
    # Strided slice
    # --------------------------------------------------------

    strided = slicer.slice(
        (
            slice(0, 4, 2),
            slice(0, 6, 2)
        )
    )

    print("\nStrided Slice")
    print("-" * 60)

    print(strided)

    expected_strided = np.array(
        [
            [0, 2, 4],
            [12, 14, 16]
        ],
        dtype=np.float64
    )

    if np.array_equal(
        strided,
        expected_strided
    ):
        print(
            "Strided slicing verification : PASS"
        )
    else:
        print(
            "Strided slicing verification : FAIL"
        )

    print("\n" + "=" * 60)
    print("SLICING TEST COMPLETED")
    print("=" * 60)