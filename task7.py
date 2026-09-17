"""
task7.py
----------------------------------------------------
Task 7: Custom Matrix Slicing & Broadcast Mechanics

Final integration and verification program.

Demonstrates:

    - Explicit tensor strides
    - Manual flat-buffer offset calculation
    - Custom slicing without NumPy slicing
    - Custom broadcasting
    - ctypes memory inspection
    - Memory views
    - Verification against NumPy results

The actual slicing and broadcasting operations are
implemented manually.
"""

import time
import numpy as np

from tensor_engine import (
    TensorEngine,
    calculate_strides,
    coordinate_to_offset,
    offset_to_coordinate
)

from slicing import CustomSlicer

from broadcasting import BroadcastEngine


# ============================================================
# UTILITY
# ============================================================

def print_section(title):
    print()
    print("-" * 60)
    print(title)
    print("-" * 60)


# ============================================================
# TEST 1 — TENSOR STRIDES
# ============================================================

def test_strides():

    print_section("1. EXPLICIT STRIDE CALCULATION")

    shape = (2, 3, 4)

    strides = calculate_strides(shape)

    print(f"Tensor shape : {shape}")
    print(f"Strides      : {strides}")

    expected = (12, 4, 1)

    if strides == expected:

        print("Stride calculation : PASS")
        return True

    print("Stride calculation : FAIL")
    return False


# ============================================================
# TEST 2 — OFFSET CALCULATION
# ============================================================

def test_offsets():

    print_section("2. FLAT-BUFFER OFFSET CALCULATION")

    shape = (2, 3, 4)

    coordinate = (1, 2, 3)

    offset = coordinate_to_offset(
        coordinate,
        shape
    )

    recovered = offset_to_coordinate(
        offset,
        shape
    )

    print(f"Shape              : {shape}")
    print(f"Coordinate         : {coordinate}")
    print(f"Calculated offset  : {offset}")
    print(f"Recovered coordinate: {recovered}")

    if (
        offset == 23
        and recovered == coordinate
    ):

        print("Offset calculation : PASS")
        return True

    print("Offset calculation : FAIL")
    return False


# ============================================================
# TEST 3 — TENSOR ENGINE
# ============================================================

def test_tensor_engine():

    print_section("3. TENSOR ENGINE")

    data = np.arange(
        24,
        dtype=np.float64
    ).reshape(2, 3, 4)

    tensor = TensorEngine(data)

    coordinate = (1, 2, 3)

    value = tensor.get_value(
        coordinate
    )

    print(f"Shape          : {tensor.shape}")
    print(f"Strides        : {tensor.strides}")
    print(f"Data type      : {tensor.dtype}")
    print(f"Elements       : {tensor.flat.size}")
    print(f"Memory bytes   : {tensor.nbytes()}")
    print(
        f"Memory address : "
        f"{tensor.memory_address()}"
    )

    print(
        f"Value at {coordinate} : "
        f"{value}"
    )

    if value == 23.0:

        print("Tensor access : PASS")
        return True

    print("Tensor access : FAIL")
    return False


# ============================================================
# TEST 4 — CUSTOM SLICING
# ============================================================

def test_custom_slicing():

    print_section("4. CUSTOM MATRIX SLICING")

    data = np.arange(
        24,
        dtype=np.float64
    ).reshape(4, 6)

    print("Original matrix:")
    print(data)

    slicer = CustomSlicer(data)

    result = slicer.slice(
        (
            slice(1, 3),
            slice(2, 6)
        )
    )

    print("\nCustom slice:")
    print(result)

    expected = np.array(
        [
            [8.0, 9.0, 10.0, 11.0],
            [14.0, 15.0, 16.0, 17.0]
        ]
    )

    if np.array_equal(
        result,
        expected
    ):

        print(
            "\nCustom slicing : PASS"
        )

        return True

    print(
        "\nCustom slicing : FAIL"
    )

    return False


# ============================================================
# TEST 5 — STRIDED SLICING
# ============================================================

def test_strided_slicing():

    print_section("5. CUSTOM STRIDED SLICING")

    data = np.arange(
        24,
        dtype=np.float64
    ).reshape(4, 6)

    slicer = CustomSlicer(data)

    result = slicer.slice(
        (
            slice(0, 4, 2),
            slice(0, 6, 2)
        )
    )

    print("Custom strided result:")
    print(result)

    expected = np.array(
        [
            [0.0, 2.0, 4.0],
            [12.0, 14.0, 16.0]
        ]
    )

    if np.array_equal(
        result,
        expected
    ):

        print(
            "Strided slicing : PASS"
        )

        return True

    print(
        "Strided slicing : FAIL"
    )

    return False


# ============================================================
# TEST 6 — VECTOR BROADCASTING
# ============================================================

def test_vector_broadcasting():

    print_section("6. VECTOR BROADCASTING")

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

    print(f"Matrix shape : {matrix.shape}")
    print(f"Vector shape : {vector.shape}")
    print(f"Result shape : {result.shape}")

    print("\nCustom broadcast result:")
    print(result)

    expected = np.array(
        [
            [11.0, 22.0, 33.0],
            [14.0, 25.0, 36.0]
        ]
    )

    if np.array_equal(
        result,
        expected
    ):

        print(
            "\nVector broadcasting : PASS"
        )

        return True

    print(
        "\nVector broadcasting : FAIL"
    )

    return False


# ============================================================
# TEST 7 — COLUMN BROADCASTING
# ============================================================

def test_column_broadcasting():

    print_section("7. COLUMN BROADCASTING")

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

    print(f"Matrix shape : {matrix.shape}")
    print(f"Column shape : {column.shape}")

    print("\nCustom result:")
    print(result)

    expected = np.array(
        [
            [11.0, 12.0, 13.0],
            [24.0, 25.0, 26.0]
        ]
    )

    if np.array_equal(
        result,
        expected
    ):

        print(
            "\nColumn broadcasting : PASS"
        )

        return True

    print(
        "\nColumn broadcasting : FAIL"
    )

    return False


# ============================================================
# TEST 8 — SCALAR BROADCASTING
# ============================================================

def test_scalar_broadcasting():

    print_section("8. SCALAR BROADCASTING")

    matrix = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0]
        ]
    )

    scalar = np.array(
        10.0
    )

    engine = BroadcastEngine(
        matrix,
        scalar
    )

    result = engine.add()

    print(f"Matrix shape : {matrix.shape}")
    print(f"Scalar shape : {scalar.shape}")

    print("\nCustom result:")
    print(result)

    expected = np.array(
        [
            [11.0, 12.0],
            [13.0, 14.0]
        ]
    )

    if np.array_equal(
        result,
        expected
    ):

        print(
            "\nScalar broadcasting : PASS"
        )

        return True

    print(
        "\nScalar broadcasting : FAIL"
    )

    return False


# ============================================================
# TEST 9 — BROADCAST MULTIPLICATION
# ============================================================

def test_broadcast_multiplication():

    print_section(
        "9. BROADCAST MULTIPLICATION"
    )

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

    print("Custom multiplication result:")
    print(result)

    expected = np.array(
        [
            [2.0, 6.0, 12.0],
            [8.0, 15.0, 24.0]
        ]
    )

    if np.array_equal(
        result,
        expected
    ):

        print(
            "\nBroadcast multiplication : PASS"
        )

        return True

    print(
        "\nBroadcast multiplication : FAIL"
    )

    return False


# ============================================================
# TEST 10 — MEMORY VIEW AND CTYPES
# ============================================================

def test_memory():

    print_section(
        "10. MEMORY VIEW & CTYPES"
    )

    data = np.arange(
        12,
        dtype=np.float64
    ).reshape(3, 4)

    tensor = TensorEngine(data)

    address = tensor.memory_address()

    memory_bytes = tensor.nbytes()

    print(
        f"Memory address : {address}"
    )

    print(
        f"Memory bytes   : {memory_bytes}"
    )

    print(
        f"Memory view format : "
        f"{tensor.memory.format}"
    )

    print(
        f"Memory view itemsize : "
        f"{tensor.memory.itemsize}"
    )

    if (
        address is not None
        and memory_bytes == 96
    ):

        print(
            "\nMemory inspection : PASS"
        )

        return True

    print(
        "\nMemory inspection : FAIL"
    )

    return False


# ============================================================
# TEST 11 — INVALID BROADCAST
# ============================================================

def test_invalid_broadcast():

    print_section(
        "11. INVALID BROADCAST SHAPE"
    )

    left = np.zeros(
        (2, 3)
    )

    right = np.zeros(
        (4, 5)
    )

    try:

        BroadcastEngine(
            left,
            right
        ).add()

        print(
            "Invalid shape handling : FAIL"
        )

        return False

    except ValueError:

        print(
            "Invalid shape handling : PASS"
        )

        return True


# ============================================================
# FINAL SUMMARY
# ============================================================

def main():

    print("=" * 60)
    print(
        "TASK 7: CUSTOM MATRIX SLICING & "
        "BROADCAST MECHANICS"
    )
    print("=" * 60)

    start_time = time.perf_counter()

    results = []

    results.append(
        test_strides()
    )

    results.append(
        test_offsets()
    )

    results.append(
        test_tensor_engine()
    )

    results.append(
        test_custom_slicing()
    )

    results.append(
        test_strided_slicing()
    )

    results.append(
        test_vector_broadcasting()
    )

    results.append(
        test_column_broadcasting()
    )

    results.append(
        test_scalar_broadcasting()
    )

    results.append(
        test_broadcast_multiplication()
    )

    results.append(
        test_memory()
    )

    results.append(
        test_invalid_broadcast()
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    # ========================================================
    # FINAL VERIFICATION
    # ========================================================

    print()
    print("=" * 60)
    print("TASK 7 VERIFICATION")
    print("=" * 60)

    passed = sum(
        1 for result in results
        if result
    )

    total = len(results)

    print(
        f"Tests passed : {passed}/{total}"
    )

    print(
        f"Execution time : "
        f"{elapsed:.6f} seconds"
    )

    print()

    if passed == total:

        print(
            "TASK 7 VERIFICATION : PASS"
        )

        print()
        print(
            "All custom tensor slicing and "
            "broadcasting tests completed successfully."
        )

    else:

        print(
            "TASK 7 VERIFICATION : FAIL"
        )

        print(
            "One or more tests require attention."
        )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()