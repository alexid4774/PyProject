'''This file is currently the main plan for launching the general program. Here, input data is processed, transferred to tests, and data is transferred for report generation.'''

import csv
from pathlib import Path
from typing import Any, Callable
import numpy as np

from core import NIST_tests_functions as module
from core import report_creator as creator
from core import user_interface as interface

Bits = list[int]
TestFunction = Callable[[np.ndarray], Any]

ALPHA = 0.01
DEFAULT_OUTPUT_PATH_CSV = "results/NIST_tests_results.csv"
DEFAULT_OUTPUT_PATH_REPORT = "results"



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
        "Block Frequency": lambda bits: module.frequency_block_test(bits, block_size = 128),
        "Runs": module.runs_test,
        "Longest Run Ones": lambda bits: module.longest_run_ones_test(bits, block_size = 128),
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


def save_results(rows: list[dict[str, Any]], output_path: str = DEFAULT_OUTPUT_PATH_CSV) -> None:

    '''Saves collected NIST results to a CSV file.'''

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents = True, exist_ok = True)

    fieldnames = ["sequence_id", "test", "subtest", "p_value", "passed", "error"]

    with path.open("w", newline="", encoding = "utf-8") as file: 
        writer = csv.DictWriter(file, fieldnames = fieldnames)
        writer.writeheader()
        writer.writerows(rows)



app = interface.InputInterface()


def run_pipeline(bits, chunk_size):

    try:
        sequences = split_into_chunks(bits, chunk_size)

        app.add_log_message(f"Test sequences: {len(sequences)}")
        app.add_log_message("Running the NIST tests...")

        rows = run_all_tests(sequences, module)

        app.add_log_message("Saving the results to a CSV file...")
        save_results(rows, DEFAULT_OUTPUT_PATH_CSV)

        app.add_log_message("I'm forwarding the CSV file for analysis...")
        app.add_log_message("I'm conducting an analysis...")

        creator.generate_report(DEFAULT_OUTPUT_PATH_CSV, DEFAULT_OUTPUT_PATH_REPORT)

        app.add_log_message("All done! The report is now in the results folder!")

    except Exception as error:
        app.add_log_message(f"ERROR: {error}")

    finally:
        app.is_running = False


app.on_run = run_pipeline
app.root.mainloop()






