import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

import sv_ttk

class FileOpener(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("test")
        self.geometry("600x600")

        self.file_path_var = tk.StringVar()

        file_path_label = ttk.Label(self, text="WSI File Path:")
        file_path_label.pack(pady=10)

        file_path_entry = ttk.Entry(self, textvariable=self.file_path_var, width=30)
        file_path_entry.pack()

        browse_button = ttk.Button(self, text="Browse", command=self.browse_file)
        browse_button.pack(pady=10)

        open_button = ttk.Button(self, text="Open", command=self.open_file)
        open_button.pack()

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("WSI Files", "*.wsi")])
        self.file_path_var.set(file_path)

    def open_file(self):
        file_path = self.file_path_var.get()
        if file_path:
            print(f"Opening file: {file_path}")

if __name__ == "__main__":
    app = FileOpener()
    sv_ttk.set_theme("dark")
    app.mainloop()