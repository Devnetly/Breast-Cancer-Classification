import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import torch

import sv_ttk
from feature_extraction import transform_wsis

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

        extract_button = ttk.Button(self, text="Extract", command=self.extract_features)
        extract_button.pack(pady=10)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("WSI Files", "*.wsi")])
        self.file_path_var.set(file_path)

    def open_file(self):
        file_path = self.file_path_var.get()
        if file_path:
            print(f"Opening file: {file_path}")

    def extract_features(self):
        file_path = self.file_path_var.get()
        if file_path:
            # Set the necessary parameters for transform_wsis
            model = None  # Replace with the appropriate model instance
            patch_size = 224
            destination_folder = "path/to/destination/folder"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            num_workers = 4  # Adjust the number of workers as needed

            transform_wsis(
                model=model,
                source_path=file_path,
                patch_size=patch_size,
                destination_folder=destination_folder,
                device=device,
                num_workers=num_workers,
            )

if __name__ == "__main__":
    app = FileOpener()
    sv_ttk.set_theme("dark")
    app.mainloop()