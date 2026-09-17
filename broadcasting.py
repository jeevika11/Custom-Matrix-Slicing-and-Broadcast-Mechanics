"""
broadcasting.py
----------------------------------------------------
Task 7: Custom Broadcasting Mechanics

Implements NumPy-like broadcasting from scratch.

The actual broadcasting operation does NOT use:

    np.broadcast_to()
    np.add()
    array + array

Instead, the implementation manually calculates:

    - Broadcast shape
    - Dimension compatibility
    - Output coordinates
    - Source coordinates
    - Flat-buffer offsets
    - Broadcasted values

Required concepts:
    - Pure NumPy
    - Explicit strides
    - Flat buffers
    - ctypes
    - Memory views
"""

import ctypes
import numpy as np

from tensor_engine import TensorEngine


class BroadcastEngine:
    """
    Implements broadcasting mechanics manually.
    """

    def __init__(self, left, right):

        if not isinstance(left, np.ndarray):
            left = np.asarray(left)

        if not isinstance(right, np.ndarray):
            right = np.asarray(right)

        self.left = TensorEngine(left)
        self.right = TensorEngine(right)

    # ========================================================
    # BROADCAST SHAPE
    # ========================================================

    @staticmethod
    def calculate_broadcast_shape(
        shape_a,
        shape_b
    ):
        """
        Calculate the resulting broadcast shape.

        Broadcasting rule:

        Two dimensions are compatible when:

            1. They are equal
            OR
            2. One of them is 1

        Dimensions are compared from right to left.
        """

        shape_a = tuple(shape_a)
        shape_b = tuple(shape_b)

        max_ndim = max(
            len(shape_a),
            len(shape_b)
        )

        result_shape = []

        for position in range(1, max_ndim + 1):

            dim_a = (
                shape_a[-position]
                if position <= len(shape_a)
                else 1
            )

            dim_b = (
                shape_b[-position]
                if position <= len(shape_b)
                else 1
            )

            if dim_a == dim_b:

                result_dim = dim_a

            elif dim_a == 1:

                result_dim = dim_b

            elif dim_b == 1:

                result_dim = dim_a

            else:

                raise ValueError(
                    "Shapes cannot be broadcast together: "
                    f"{shape_a} and {shape_b}"
                )

            result_shape.append(result_dim)

        result_shape.reverse()

        return tuple(result_shape)

    # ========================================================
    # ALIGN SHAPE
    # ========================================================

    @staticmethod
    def _align_shape(
        shape,
        target_ndim
    ):
        """
        Add leading dimensions of size 1 so that
        two shapes have equal dimensionality.

        Example:

            shape = (3,)
            target = 2

            aligned = (1, 3)
        """

        missing = (
            target_ndim - len(shape)
        )

        return (
            (1,) * missing
            + tuple(shape)
        )

    # ========================================================
    # ALIGN STRIDES
    # ========================================================

    @staticmethod
    def _align_strides(
        shape,
        strides,
        target_ndim
    ):
        """
        Align strides with the broadcast dimensions.

        Newly inserted leading dimensions receive
        a stride of zero.

        A broadcast dimension of size 1 also receives
        stride zero because the same source value is reused.
        """

        aligned_shape = BroadcastEngine._align_shape(
            shape,
            target_ndim
        )

        missing = (
            target_ndim - len(shape)
        )

        aligned_strides = (
            (0,) * missing
            + tuple(strides)
        )

        adjusted = []

        for dimension, stride in zip(
            aligned_shape,
            aligned_strides
        ):

            if dimension == 1:

                adjusted.append(0)

            else:

                adjusted.append(stride)

        return tuple(adjusted)

    # ========================================================
    # OUTPUT STRIDES
    # ========================================================

    @staticmethod
    def _calculate_strides(shape):
        """
        Calculate C-order element strides.
        """

        if len(shape) == 0:
            return ()

        strides = [1] * len(shape)

        for index in range(
            len(shape) - 2,
            -1,
            -1
        ):

            strides[index] = (
                strides[index + 1]
                * shape[index + 1]
            )

        return tuple(strides)

    # ========================================================
    # OFFSET FROM OUTPUT COORDINATE
    # ========================================================

    @staticmethod
    def _source_offset(
        output_coordinate,
        aligned_shape,
        aligned_strides
    ):
        """
        Calculate the source flat-buffer offset
        for a particular output coordinate.

        Broadcasting dimensions use the same source
        element repeatedly.
        """

        offset = 0

        for coordinate, dimension, stride in zip(
            output_coordinate,
            aligned_shape,
            aligned_strides
        ):

            if dimension == 1:

                source_coordinate = 0

            else:

                source_coordinate = coordinate

            offset += (
                source_coordinate * stride
            )

        return offset

    # ========================================================
    # FLAT COORDINATE GENERATOR
    # ========================================================

    @staticmethod
    def _generate_coordinates(
        shape
    ):
        """
        Generate all multidimensional coordinates
        without using NumPy indexing.

        Example:

            shape = (2, 3)

        Generates:

            (0, 0)
            (0, 1)
            (0, 2)
            (1, 0)
            (1, 1)
            (1, 2)
        """

        if len(shape) == 0:
            yield ()
            return

        coordinate = [0] * len(shape)

        while True:

            yield tuple(coordinate)

            axis = len(shape) - 1

            while axis >= 0:

                coordinate[axis] += 1

                if coordinate[axis] < shape[axis]:
                    break

                coordinate[axis] = 0
                axis -= 1

            if axis < 0:
                break

    # ========================================================
    # CUSTOM BROADCAST ADDITION
    # ========================================================

    def add(self):
        """
        Perform element-wise addition using manually
        calculated broadcast offsets.

        No NumPy broadcasting is used.
        """

        result_shape = (
            self.calculate_broadcast_shape(
                self.left.shape,
                self.right.shape
            )
        )

        result = np.empty(
            result_shape,
            dtype=np.result_type(
                self.left.dtype,
                self.right.dtype
            )
        )

        target_ndim = len(result_shape)

        left_shape = self._align_shape(
            self.left.shape,
            target_ndim
        )

        right_shape = self._align_shape(
            self.right.shape,
            target_ndim
        )

        left_strides = self._align_strides(
            self.left.shape,
            self.left.strides,
            target_ndim
        )

        right_strides = self._align_strides(
            self.right.shape,
            self.right.strides,
            target_ndim
        )

        result_strides = self._calculate_strides(
            result_shape
        )

        # ----------------------------------------------------
        # Iterate over output coordinates.
        # ----------------------------------------------------

        for output_coordinate in self._generate_coordinates(
            result_shape
        ):

            left_offset = self._source_offset(
                output_coordinate,
                left_shape,
                left_strides
            )

            right_offset = self._source_offset(
                output_coordinate,
                right_shape,
                right_strides
            )

            result_offset = 0

            for coordinate, stride in zip(
                output_coordinate,
                result_strides
            ):

                result_offset += (
                    coordinate * stride
                )

            left_value = self.left.flat[
                left_offset
            ]

            right_value = self.right.flat[
                right_offset
            ]

            result.flat[
                result_offset
            ] = left_value + right_value

        return result

    # ========================================================
    # CUSTOM BROADCAST MULTIPLICATION
    # ========================================================

    def multiply(self):
        """
        Perform element-wise multiplication using
        manually calculated broadcast offsets.
        """

        result_shape = (
            self.calculate_broadcast_shape(
                self.left.shape,
                self.right.shape
            )
        )

        result = np.empty(
            result_shape,
            dtype=np.result_type(
                self.left.dtype,
                self.right.dtype
            )
        )

        target_ndim = len(result_shape)

        left_shape = self._align_shape(
            self.left.shape,
            target_ndim
        )

        right_shape = self._align_shape(
            self.right.shape,
            target_ndim
        )

        left_strides = self._align_strides(
            self.left.shape,
            self.left.strides,
            target_ndim
        )

        right_strides = self._align_strides(
            self.right.shape,
            self.right.strides,
            target_ndim
        )

        result_strides = self._calculate_strides(
            result_shape
        )

        for output_coordinate in self._generate_coordinates(
            result_shape
        ):

            left_offset = self._source_offset(
                output_coordinate,
                left_shape,
                left_strides
            )

            right_offset = self._source_offset(
                output_coordinate,
                right_shape,
                right_strides
            )

            result_offset = 0

            for coordinate, stride in zip(
                output_coordinate,
                result_strides
            ):

                result_offset += (
                    coordinate * stride
                )

            result.flat[
                result_offset
            ] = (
                self.left.flat[left_offset]
                * self.right.flat[right_offset]
            )

        return result

    # ========================================================
    # MEMORY INFORMATION
    # ========================================================

    def memory_information(self):
        """
        Return memory information for both tensors.
        """

        left_address = (
            ctypes.addressof(
                ctypes.c_char.from_buffer(
                    self.left.memory
                )
            )
            if self.left.flat.size > 0
            else None
        )

        right_address = (
            ctypes.addressof(
                ctypes.c_char.from_buffer(
                    self.right.memory
                )
            )
            if self.right.flat.size > 0
            else None
        )

        return {
            "left_address": left_address,
            "right_address": right_address,
            "left_bytes": self.left.nbytes(),
            "right_bytes": self.right.nbytes()
        }


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def custom_broadcast_add(
    left,
    right
):
    """
    Convenience wrapper for custom broadcasting addition.
    """

    engine = BroadcastEngine(
        left,
        right
    )

    return engine.add()


# ============================================================
# TEST 1 — VECTOR BROADCASTING
# ============================================================

def test_vector_broadcasting():

    print("\n" + "-" * 60)
    print("TEST 1: VECTOR BROADCASTING")
    print("-" * 60)

    matrix = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]
    )

    vector = np.array(
        [10.0, 20.0, 30.0]
    )

    engine = BroadcastEngine(
        matrix,
        vector
    )

    result = engine.add()

    expected = np.array(
        [
            [11.0, 22.0, 33.0],
            [14.0, 25.0, 36.0]
        ]
    )

    print("Matrix shape :", matrix.shape)
    print("Vector shape :", vector.shape)
    print("Result shape :", result.shape)

    print("\nCustom result:")
    print(result)

    if np.array_equal(
        result,
        expected
    ):
        print(
            "\nVector broadcasting : PASS"
        )
    else:
        print(
            "\nVector broadcasting : FAIL"
        )


# ============================================================
# TEST 2 — COLUMN BROADCASTING
# ============================================================

def test_column_broadcasting():

    print("\n" + "-" * 60)
    print("TEST 2: COLUMN BROADCASTING")
    print("-" * 60)

    matrix = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]
    )

    column = np.array(
        [
            [10.0],
            [20.0]
        ]
    )

    engine = BroadcastEngine(
        matrix,
        column
    )

    result = engine.add()

    expected = np.array(
        [
            [11.0, 12.0, 13.0],
            [24.0, 25.0, 26.0]
        ]
    )

    print("Matrix shape :", matrix.shape)
    print("Column shape :", column.shape)
    print("Result shape :", result.shape)

    print("\nCustom result:")
    print(result)

    if np.array_equal(
        result,
        expected
    ):
        print(
            "\nColumn broadcasting : PASS"
        )
    else:
        print(
            "\nColumn broadcasting : FAIL"
        )


# ============================================================
# TEST 3 — SCALAR BROADCASTING
# ============================================================

def test_scalar_broadcasting():

    print("\n" + "-" * 60)
    print("TEST 3: SCALAR BROADCASTING")
    print("-" * 60)

    matrix = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0]
        ]
    )

    scalar = np.array(10.0)

    engine = BroadcastEngine(
        matrix,
        scalar
    )

    result = engine.add()

    expected = np.array(
        [
            [11.0, 12.0],
            [13.0, 14.0]
        ]
    )

    print("Matrix shape :", matrix.shape)
    print("Scalar shape :", scalar.shape)
    print("Result shape :", result.shape)

    print("\nCustom result:")
    print(result)

    if np.array_equal(
        result,
        expected
    ):
        print(
            "\nScalar broadcasting : PASS"
        )
    else:
        print(
            "\nScalar broadcasting : FAIL"
        )


# ============================================================
# TEST 4 — MULTIPLICATION
# ============================================================

def test_multiplication():

    print("\n" + "-" * 60)
    print("TEST 4: BROADCAST MULTIPLICATION")
    print("-" * 60)

    matrix = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]
    )

    vector = np.array(
        [2.0, 3.0, 4.0]
    )

    engine = BroadcastEngine(
        matrix,
        vector
    )

    result = engine.multiply()

    expected = np.array(
        [
            [2.0, 6.0, 12.0],
            [8.0, 15.0, 24.0]
        ]
    )

    print("Custom result:")
    print(result)

    if np.array_equal(
        result,
        expected
    ):
        print(
            "\nBroadcast multiplication : PASS"
        )
    else:
        print(
            "\nBroadcast multiplication : FAIL"
        )


# ============================================================
# TEST 5 — MEMORY INFORMATION
# ============================================================

def test_memory():

    print("\n" + "-" * 60)
    print("TEST 5: MEMORY INFORMATION")
    print("-" * 60)

    left = np.arange(
        6,
        dtype=np.float64
    ).reshape(2, 3)

    right = np.array(
        [1.0, 2.0, 3.0]
    )

    engine = BroadcastEngine(
        left,
        right
    )

    information = (
        engine.memory_information()
    )

    print(
        f"Left memory address  : "
        f"{information['left_address']}"
    )

    print(
        f"Right memory address : "
        f"{information['right_address']}"
    )

    print(
        f"Left memory bytes    : "
        f"{information['left_bytes']}"
    )

    print(
        f"Right memory bytes   : "
        f"{information['right_bytes']}"
    )

    if (
        information["left_address"] is not None
        and information["right_address"] is not None
    ):
        print(
            "\nMemory inspection : PASS"
        )
    else:
        print(
            "\nMemory inspection : FAIL"
        )


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CUSTOM BROADCASTING ENGINE TEST")
    print("=" * 60)

    test_vector_broadcasting()

    test_column_broadcasting()

    test_scalar_broadcasting()

    test_multiplication()

    test_memory()

    print("\n" + "=" * 60)
    print("BROADCASTING TEST COMPLETED")
    print("=" * 60)