'''This file contains a graphical interface for selecting input data for NIST tests.'''

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import random
import threading
import sys

Bits = list[int]


class InputInterface:

    '''Graphical interface for selecting bit sequence input method.'''

    def __init__(self) -> None:

        self.is_running = False
        self.on_run = None
        self.root = tk.Tk()
        self.root.title("NIST tests input interface")
        self.root.geometry("850x680")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.bits: Bits | None = None
        self.chunk_size: int | None = None
        self.is_ready = False

        self.input_mode = tk.StringVar(value = "keyboard")
        self.file_path = tk.StringVar(value = "")
        self.random_length = tk.StringVar(value = "10000")
        self.chunk_size_value = tk.StringVar(value = "")

        self.status_text = tk.StringVar(value = "Select a method for transmitting the sequence.")
        self.log_messages: list[str] = []

        self.create_widgets()


    def create_widgets(self) -> None:

        '''Creates all interface widgets.'''

        title = tk.Label(self.root, text = "NIST Randomness Tester", font = ("Arial", 22, "bold"))
        title.pack(pady = 15)

        mode_frame = tk.LabelFrame(self.root, text = "Input method", padx = 15, pady = 10)
        mode_frame.pack(fill = "x", padx = 20)

        tk.Radiobutton(mode_frame, text = "Keyboard input", variable = self.input_mode, value = "keyboard",
                       command = self.update_mode).pack(anchor = "w")

        tk.Radiobutton(mode_frame, text = "Load from file", variable = self.input_mode, value = "file",
                       command = self.update_mode).pack(anchor = "w")

        tk.Radiobutton(mode_frame, text = "Generating a bit sequence with module random", variable = self.input_mode,
                       value = "random", command = self.update_mode).pack(anchor = "w")

        self.input_area = tk.Frame(self.root)
        self.input_area.pack(fill = "x", padx = 20, pady = 10)

        self.keyboard_frame = tk.Frame(self.input_area)

        self.keyboard_text = tk.Text(self.keyboard_frame, height = 6, width = 90)
        self.keyboard_text.pack(pady = 5)

        keyboard_buttons_frame = tk.Frame(self.keyboard_frame)
        keyboard_buttons_frame.pack(pady = 5)

        tk.Button(keyboard_buttons_frame, text = "0", width = 10, command = lambda: self.add_bit("0")).pack(side = "left", padx = 5)
        tk.Button(keyboard_buttons_frame, text = "1", width = 10, command = lambda: self.add_bit("1")).pack(side = "left", padx = 5)
        tk.Button(keyboard_buttons_frame, text = "Clear", width = 14, command = self.clear_keyboard_input).pack(side = "left", padx = 5)

        self.file_frame = tk.Frame(self.input_area)
        tk.Entry(self.file_frame, textvariable = self.file_path, width = 70).pack(side = "left", padx = 10)
        tk.Button(self.file_frame, text = "Select a file", command = self.choose_file).pack(side = "left")

        self.random_frame = tk.Frame(self.input_area)
        tk.Label(self.random_frame, text = "Sequence length:").pack(side = "left", padx = 10)
        tk.Entry(self.random_frame, textvariable = self.random_length, width = 15).pack(side = "left")

        settings_frame = tk.LabelFrame(self.root, text = "Settings", padx = 15, pady = 10)
        settings_frame.pack(fill = "x", padx = 20, pady = 10)

        tk.Label(settings_frame, text = "Block size for splitting, enter or leave blank to skip splitting:").pack(side = "left")
        tk.Entry(settings_frame, textvariable = self.chunk_size_value, width = 15).pack(side = "left", padx = 10)

        self.status_label = tk.Label(self.root, textvariable = self.status_text, fg = "white", font = ("Arial", 11))
        self.status_label.pack(pady = 5)

        buttons_frame = tk.Frame(self.root)
        buttons_frame.pack(pady = 10)

        tk.Button(buttons_frame, text = "Run", width = 18, command = self.submit).pack(side = "left", padx = 10)
        tk.Button(buttons_frame, text = "Exit", width = 18, command = self.close).pack(side = "left", padx = 10)
        tk.Button(buttons_frame, text = "Instruction", width = 18, command = self.show_instruction).pack(side = "left", padx = 10)

        self.log_text = tk.Text(self.root, height = 10, width = 90, state = "disabled")
        self.log_text.pack(padx = 20, pady = 10)

        self.update_mode()


    def show_instruction(self) -> None:

        '''Shows user instruction window.'''

        instruction_text = (
            "NIST Randomness Tester Instruction\n\n"
            "1. Select input mode:\n"
            "   - Keyboard input\n"
            "   - Text file\n"
            "   - Random sequence generation\n\n"
            "2. The program reads only bit values: 0 and 1.\n"
            "   All characters except 0 and 1 are ignored.\n\n"
            "3. If you use a file, it may contain spaces, new lines or other separators.\n"
            "   They will be ignored automatically.\n\n"
            "4. Chunk size is optional.\n"
            "   If chunk size is empty, the whole sequence is tested as one sequence.\n"
            "   If chunk size is set, the sequence is split into equal blocks.\n\n"
            "5. After pressing Run, the program will:\n"
            "   - read the bit sequence;\n"
            "   - run NIST tests;\n"
            "   - save CSV results;\n"
            "   - generate a report with charts.\n")

        messagebox.showinfo("Instruction", instruction_text)


    def update_mode(self) -> None:

        '''Shows only widgets required for selected input mode.'''

        self.keyboard_frame.pack_forget()
        self.file_frame.pack_forget()
        self.random_frame.pack_forget()

        if self.input_mode.get() == "keyboard":
            self.keyboard_frame.pack(fill = "x")
            self.status_text.set("Enter the bit sequence manually.")

        elif self.input_mode.get() == "file":
            self.file_frame.pack(fill = "x")
            self.status_text.set("Select a text file containing a bit sequence.")

        elif self.input_mode.get() == "random":
            self.random_frame.pack(fill = "x")
            self.status_text.set("Specify the length of the random sequence.")


    def choose_file(self) -> None:

        '''Opens file selection dialog.'''

        path = filedialog.askopenfilename(title = "Select a file containing bits",
                                          filetypes = [("Text files", "*.txt"), ("All files", "*.*")])

        if path:
            self.file_path.set(path)
            self.status_text.set(f"File selected: {path}")


    def load_bits_from_file(self, path: str) -> Bits:

        '''Reads a text file and extracts only 0/1 characters as bits.'''

        file_path = Path(path).expanduser().resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        text = file_path.read_text(encoding = "utf-8")
        bits = [int(ch) for ch in text if ch in "01"]

        if not bits:
            raise ValueError("No bits were found in the selected file.")

        return bits


    def load_bits_from_keyboard(self) -> Bits:

        '''Reads bit sequence from text field and keeps only 0/1 characters.'''

        text = self.keyboard_text.get("1.0", tk.END)
        bits = [int(ch) for ch in text if ch in "01"]

        if not bits:
            raise ValueError("No bits were entered.")

        return bits


    def generate_random_bits(self) -> Bits:

        '''Generates random bit sequence using Python random library.'''

        length = int(self.random_length.get())

        if length <= 0:
            raise ValueError("Random sequence length must be positive.")

        return [random.randint(0, 1) for i in range(length)]


    def get_chunk_size(self) -> int | None:

        '''Reads chunk_size value from interface.'''

        value = self.chunk_size_value.get().strip()

        if not value:
            return None

        chunk_size = int(value)

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

        return chunk_size


    def submit(self) -> None:

        '''Processes selected input mode and starts main processing in background thread.'''

        if self.is_running:
            messagebox.showwarning("Warning!", "The program is already running.")
            return

        try:
            self.add_log_message("Processing input data...")

            if self.input_mode.get() == "keyboard":
                self.bits = self.load_bits_from_keyboard()

            elif self.input_mode.get() == "file":
                self.bits = self.load_bits_from_file(self.file_path.get())

            elif self.input_mode.get() == "random":
                self.bits = self.generate_random_bits()

            self.chunk_size = self.get_chunk_size()
            self.is_running = True

            self.add_log_message(f"Done! Bits received: {len(self.bits)}")

            if self.on_run is not None:
                thread = threading.Thread(target = self.on_run, args = (self.bits, self.chunk_size), daemon = True)
                thread.start()

        except Exception as error:
            self.is_running = False
            self.status_text.set("An error occurred while processing the data!")
            messagebox.showerror("Error!", str(error))


    def run(self) -> tuple[Bits, int | None]:

        '''Starts interface and returns selected bits and chunk_size.'''

        self.root.mainloop()

        if self.bits is None:
            raise ValueError("Input was not selected.")

        return self.bits, self.chunk_size


    def add_bit(self, bit: str) -> None:

        '''Adds selected bit to keyboard input field.'''

        self.keyboard_text.insert(tk.END, bit)
        self.status_text.set(f"A bit has been added: {bit}")


    def clear_keyboard_input(self) -> None:

        '''Clears keyboard input field.'''

        self.keyboard_text.delete("1.0", tk.END)
        self.status_text.set("The input field has been cleared.")


    def add_log_message(self, message: str) -> None:

        '''Adds message to interface log field.'''

        self.log_messages.append(message)

        self.log_text.config(state = "normal")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state = "disabled")

        self.status_text.set(message)
        self.root.update_idletasks()


    def close(self) -> None:

        '''Closes interface window and stops program execution.'''

        self.root.destroy()
        sys.exit(0)


def launch_input_interface() -> tuple[Bits, int | None]:

    '''Launches graphical input interface and returns bits with chunk_size.'''

    interface = InputInterface()
    return interface.run()