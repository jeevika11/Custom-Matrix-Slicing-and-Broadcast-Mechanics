# Task 7 – Custom Matrix Slicing & Broadcast Mechanics in Pure NumPy

## 📌 Overview

This project implements a custom tensor projection and broadcasting engine using
low-level NumPy memory concepts.

The objective is to understand how multidimensional arrays are represented in
memory and how operations such as slicing and broadcasting can be implemented
from first principles.

The implementation avoids relying on NumPy's standard slicing and broadcasting
operators for the core operations.

---

## 🎯 Objectives

The project demonstrates:

- Explicit memory stride calculation
- Multidimensional coordinate conversion
- Flat-buffer offset calculation
- Custom matrix slicing
- Custom strided slicing
- NumPy-like broadcasting mechanics
- Vector broadcasting
- Column broadcasting
- Scalar broadcasting
- Broadcast multiplication
- `ctypes` memory-address inspection
- Python memory views
- Validation against expected NumPy results

---

## 🛠️ Required Technology

- Python 3.10+
- NumPy
- ctypes
- Memory Views

---

## 📂 Project Structure

```text
Task 7/
│
├── tensor_engine.py
├── slicing.py
├── broadcasting.py
├── task7.py
├── requirements.txt
├── README.md
└── .gitignore