'''This file is currently the main plan for launching the general program. Here, input data is processed, transferred to tests, and data is transferred for report generation.'''

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from core import NIST_tests_functions as module

import csv
from pathlib import Path
from typing import Any, Callable

import numpy as np


Bits = list[int]
TestFunction = Callable[[np.ndarray], Any]


ALPHA = 0.01
DEFAULT_OUTPUT_PATH = "results/NIST_tests_results.csv"


def load_bits_from_file(path: str) -> Bits:

    '''Reads a text file and extracts only 0/1 characters as bits.'''

    file_path = Path(path).expanduser().resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = file_path.read_text(encoding = "utf-8")
    bits = [int(ch) for ch in text if ch in "01"]

    if not bits:
        raise ValueError("No bits were found in the selected file.")

    return bits


def load_bits_from_keyboard() -> Bits:

    '''Reads a bit sequence from keyboard input and keeps only 0/1 characters.'''

    text = input("Введите битовую последовательность из 0 и 1: ")
    bits = [int(ch) for ch in text if ch in "01"]

    if not bits:
        raise ValueError("No bits were entered.")

    return bits


def ask_input_source() -> Bits:

    '''Asks the user whether to read bits from a file or from keyboard input.'''

    while True:
        print("\nЧто подать на вход?")
        print("1 - текстовый файл")
        print("2 - ввод с клавиатуры")

        choice = input("Ваш выбор [1/2]: ").strip()

        if choice == "1":
            path = input("Введите путь к файлу: ").strip().strip('"').strip("'")
            return load_bits_from_file(path)

        if choice == "2":
            return load_bits_from_keyboard()

        print("Неверный выбор. Введите 1 или 2.")


def split_into_chunks(bits: Bits, chunk_size: int | None) -> list[np.ndarray]:

    '''Splits one long bit sequence into chunks or returns it as one sequence.'''

    if chunk_size is None or chunk_size <= 0:
        return [np.asarray(bits, dtype = np.uint8)]

    chunks = []
    for start in range(0, len(bits), chunk_size):
        chunk = bits[start:start + chunk_size]
        if len(chunk) == chunk_size:
            chunks.append(np.asarray(chunk, dtype = np.uint8))

    if not chunks:
        raise ValueError("Input is shorter than chunk_size.")

    return chunks


def get_tests(module: Any) -> dict[str, TestFunction]:

    '''Returns all NIST test functions that should be executed.'''
    
    return {
        "Monobit": module.frequency_monobit_test,
        "Block Frequency": lambda bits: module.frequency_block_test(bits, block_size=128),
        "Runs": module.runs_test,
        "Longest Run Ones": lambda bits: module.longest_run_ones_test(bits, block_size=128),
        "Binary Matrix Rank": module.binary_matrix_rank_test,
        "DFT": module.discrete_fourier_transform_test,
        "Non-overlapping Template": module.non_overlapping_template_test,
        "Overlapping Template": module.overlapping_template_test,
        "Universal": module.universal_statistical_test,
        "Linear Complexity": module.linear_complexity_test,
        "Serial": module.serial_test,
        "Approximate Entropy": module.approximate_entropy_test,
        "Cumulative Sums Forward": lambda bits: module.cumulative_sums_test(bits, mode = "forward"),
        "Cumulative Sums Backward": lambda bits: module.cumulative_sums_test(bits, mode = "backward"),
        "Random Excursions": module.random_excursions_test,
        "Random Excursions Variant": module.random_excursions_variant_test}


def add_result_row(rows: list[dict[str, Any]], sequence_id: int, test_name: str, result: Any) -> None:

    '''Converts a float or dict result into CSV-ready result rows.'''

    if isinstance(result, dict):
        for subtest, p_value in result.items():
            rows.append({"sequence_id": sequence_id, "test": test_name, "subtest": str(subtest), "p_value": float(p_value), 
                         "passed": float(p_value) >= ALPHA, "error": ""})
    else:
        rows.append({"sequence_id": sequence_id, "test": test_name, "subtest": "", "p_value": float(result), 
                     "passed": float(result) >= ALPHA, "error": ""})


def run_all_tests(sequences: list[np.ndarray], module: Any) -> list[dict[str, Any]]:

    '''Runs every NIST test for every sequence and returns result rows.'''

    tests = get_tests(module)
    rows: list[dict[str, Any]] = []

    for sequence_id, bits in enumerate(sequences, start=1):
        for test_name, test_func in tests.items():
            try:
                result = test_func(bits)
                add_result_row(rows, sequence_id, test_name, result)
            except Exception as error:
                rows.append({"sequence_id": sequence_id, "test": test_name, "subtest": "", "p_value": "", "passed": "",
                    "error": str(error)})

    return rows


def save_results(rows: list[dict[str, Any]], output_path: str = DEFAULT_OUTPUT_PATH) -> None:

    '''Saves collected NIST results to a CSV file.'''

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents = True, exist_ok = True)

    fieldnames = ["sequence_id", "test", "subtest", "p_value", "passed", "error"]

    with path.open("w", newline="", encoding = "utf-8") as file: 
        writer = csv.DictWriter(file, fieldnames = fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nРезультаты сохранены: {path}")


def ask_chunk_size() -> int | None:

    '''Asks whether the input sequence should be split into equal chunks.'''

    value = input("\nРазмер блока для разбиения? Enter - не разбивать: ").strip()

    if not value:
        return None

    chunk_size = int(value)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    return chunk_size


bits = ask_input_source()
chunk_size = ask_chunk_size()


output_path = input(f"Куда сохранить CSV? Enter - {DEFAULT_OUTPUT_PATH}: ").strip()
if not output_path:
    output_path = DEFAULT_OUTPUT_PATH

sequences = split_into_chunks(bits, chunk_size)

print(f"\nПоследовательностей для проверки: {len(sequences)}")
print("Запускаю NIST-тесты...")

rows = run_all_tests(sequences, module)
save_results(rows, output_path)

print("Готово. Теперь можно передать CSV-файлы дальше для анализа!")





