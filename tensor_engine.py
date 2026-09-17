"""
tensor_engine.py
----------------------------------------------------
Task 7: Custom Matrix Slicing & Broadcast Mechanics

Core tensor utilities for:

    - Explicit stride calculation
    - Flat-buffer offset calculation
    - Multidimensional coordinate conversion
    - ctypes memory inspection
    - Memory-view based access

Standard NumPy slicing is NOT used for the
actual indexing calculations.
"""

import ctypes
import numpy as np


class TensorEngine:
    """
    Low-level tensor representation.

    The tensor is represented using:
        - NumPy array for input storage
        - 1D flat buffer
        - Explicit shape
        - Explicit element strides
        - Memory view
    """

    def __init__(self, array):
        if not isinstance(array, np.ndarray):
            array = np.asarray(array)

        if array.ndim == 0:
            array = array.reshape((1,))

        # Make the array contiguous so that our manually
        # calculated row-major offsets are predictable.
        self.array = np.ascontiguousarray(array)

        self.shape = tuple(self.array.shape)
        self.ndim = self.array.ndim
        self.dtype = self.array.dtype

        # Element strides rather than byte strides.
        self.strides = self._calculate_strides(self.shape)

        # Flat 1D representation.
        self.flat = self.array.reshape(-1)

        # Memory view over the underlying data.
        self.memory = memoryview(self.flat)

    # ========================================================
    # STRIDE CALCULATION
    # ========================================================

    @staticmethod
    def _calculate_strides(shape):
        """
        Calculate C-order element strides.

        Example:

            shape = (3, 4)

            strides = (4, 1)

        For:

            shape = (2, 3, 4)

            strides = (12, 4, 1)
        """

        if len(shape) == 0:
            return ()

        strides = [1] * len(shape)

        for i in range(len(shape) - 2, -1, -1):
            strides[i] = (
                strides[i + 1] * shape[i + 1]
            )

        return tuple(strides)

    # ========================================================
    # COORDINATE VALIDATION
    # ========================================================

    def _validate_coordinates(self, coordinates):
        """
        Validate multidimensional coordinates.
        """

        if len(coordinates) != self.ndim:
            raise ValueError(
                f"Expected {self.ndim} coordinates, "
                f"got {len(coordinates)}"
            )

        for axis, coordinate in enumerate(coordinates):

            if not isinstance(coordinate, (int, np.integer)):
                raise TypeError(
                    "Coordinates must be integers."
                )

            if coordinate < 0 or coordinate >= self.shape[axis]:
                raise IndexError(
                    f"Coordinate {coordinate} is out of "
                    f"bounds for axis {axis} with size "
                    f"{self.shape[axis]}"
                )

    # ========================================================
    # OFFSET CALCULATION
    # ========================================================

    def coordinate_to_offset(self, coordinates):
        """
        Convert an N-dimensional coordinate into
        a flat-buffer element offset.

        Formula:

            offset =
                c0 * stride0 +
                c1 * stride1 +
                ...
        """

        coordinates = tuple(coordinates)

        self._validate_coordinates(coordinates)

        offset = 0

        for coordinate, stride in zip(
            coordinates,
            self.strides
        ):
            offset += coordinate * stride

        return offset

    # ========================================================
    # OFFSET TO COORDINATE
    # ========================================================

    def offset_to_coordinate(self, offset):
        """
        Convert a flat element offset back into
        an N-dimensional coordinate.
        """

        if offset < 0 or offset >= self.flat.size:
            raise IndexError(
                f"Offset {offset} is outside "
                f"the tensor buffer."
            )

        coordinates = [0] * self.ndim
        remaining = int(offset)

        for axis in range(self.ndim):

            stride = self.strides[axis]

            coordinates[axis] = (
                remaining // stride
            )

            remaining %= stride

        return tuple(coordinates)

    # ========================================================
    # GET VALUE
    # ========================================================

    def get_value(self, coordinates):
        """
        Get a tensor value using manually calculated
        flat-buffer offset.

        No multidimensional NumPy indexing is used.
        """

        offset = self.coordinate_to_offset(
            coordinates
        )

        return self.flat[offset].item()

    # ========================================================
    # SET VALUE
    # ========================================================

    def set_value(self, coordinates, value):
        """
        Set a tensor value using a manually calculated
        flat-buffer offset.
        """

        offset = self.coordinate_to_offset(
            coordinates
        )

        self.flat[offset] = value

    # ========================================================
    # FLAT BUFFER
    # ========================================================

    def get_flat_value(self, offset):
        """
        Read directly from the 1D flat buffer.
        """

        if offset < 0 or offset >= self.flat.size:
            raise IndexError(
                "Flat-buffer offset out of range."
            )

        return self.flat[offset].item()

    # ========================================================
    # MEMORY ADDRESS
    # ========================================================

    def memory_address(self):
        """
        Return the memory address of the tensor buffer
        using ctypes.
        """

        if self.flat.size == 0:
            return None

        return ctypes.addressof(
            ctypes.c_char.from_buffer(
                self.memory
            )
        )

    # ========================================================
    # BYTE INFORMATION
    # ========================================================

    def nbytes(self):
        """
        Return total number of bytes occupied by
        the tensor data.
        """

        return self.flat.nbytes

    # ========================================================
    # DESCRIPTION
    # ========================================================

    def describe(self):
        """
        Return tensor metadata.
        """

        return {
            "shape": self.shape,
            "ndim": self.ndim,
            "dtype": str(self.dtype),
            "strides": self.strides,
            "elements": int(self.flat.size),
            "bytes": int(self.nbytes()),
            "memory_address": self.memory_address()
        }


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_strides(shape):
    """
    Standalone stride calculation function.
    """

    return TensorEngine._calculate_strides(shape)


def coordinate_to_offset(
    coordinates,
    shape
):
    """
    Standalone coordinate-to-offset calculation.
    """

    strides = calculate_strides(shape)

    if len(coordinates) != len(shape):
        raise ValueError(
            "Coordinate dimensionality does not "
            "match tensor shape."
        )

    offset = 0

    for coordinate, dimension, stride in zip(
        coordinates,
        shape,
        strides
    ):

        if coordinate < 0 or coordinate >= dimension:
            raise IndexError(
                "Coordinate out of bounds."
            )

        offset += coordinate * stride

    return offset


def offset_to_coordinate(
    offset,
    shape
):
    """
    Standalone flat-offset-to-coordinate conversion.
    """

    total_elements = 1

    for dimension in shape:
        total_elements *= dimension

    if offset < 0 or offset >= total_elements:
        raise IndexError(
            "Offset out of bounds."
        )

    strides = calculate_strides(shape)

    coordinates = []
    remaining = offset

    for stride in strides:

        coordinate = remaining // stride
        coordinates.append(coordinate)

        remaining %= stride

    return tuple(coordinates)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TENSOR ENGINE TEST")
    print("=" * 60)

    data = np.arange(
        24,
        dtype=np.float64
    ).reshape(2, 3, 4)

    tensor = TensorEngine(data)

    print("\nTensor Information")
    print("-" * 60)

    info = tensor.describe()

    for key, value in info.items():
        print(f"{key:<18}: {value}")

    # --------------------------------------------------------
    # Test coordinate -> offset
    # --------------------------------------------------------

    coordinate = (1, 2, 3)

    offset = tensor.coordinate_to_offset(
        coordinate
    )

    print("\nCoordinate to Offset")
    print("-" * 60)
    print(f"Coordinate : {coordinate}")
    print(f"Offset     : {offset}")
    print(
        f"Value      : "
        f"{tensor.get_value(coordinate)}"
    )

    # --------------------------------------------------------
    # Test offset -> coordinate
    # --------------------------------------------------------

    recovered = tensor.offset_to_coordinate(
        offset
    )

    print("\nOffset to Coordinate")
    print("-" * 60)
    print(f"Offset     : {offset}")
    print(f"Coordinate : {recovered}")

    # --------------------------------------------------------
    # Memory information
    # --------------------------------------------------------

    print("\nMemory Information")
    print("-" * 60)

    print(
        f"Memory address : "
        f"{tensor.memory_address()}"
    )

    print(
        f"Memory bytes   : "
        f"{tensor.nbytes()}"
    )

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    if recovered == coordinate:
        print("\nCoordinate conversion : PASS")
    else:
        print("\nCoordinate conversion : FAIL")

    if tensor.get_value(coordinate) == 23.0:
        print("Flat-buffer access   : PASS")
    else:
        print("Flat-buffer access   : FAIL")

    print("\n" + "=" * 60)
    print("TENSOR ENGINE TEST COMPLETED")
    print("=" * 60)