import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import torch
from threading import Thread
from feature_extraction import transform_wsis

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

        self.progress_var = tk.StringVar()
        progress_label = ttk.Label(self, textvariable=self.progress_var)
        progress_label.pack(pady=10)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("WSI Files", "*.svs")])
        self.file_path_var.set(file_path)

    def open_file(self):
        file_path = self.file_path_var.get()
        if file_path:
            thread = Thread(target=transform_wsis, kwargs={
                'model': None,
                'source_path': file_path,
                'patch_size': 224,
                'destination_folder': '/path/to/destination/folder',
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'prefetch_factor': 2,
                'app': self
            })
            thread.start()

    def update_progress(self, message):
        def update():
            self.progress_var.set(message)
            self.update_idletasks()

        self.after(0, update)

if __name__ == "__main__":
    app = FileOpener()
    sv_ttk.set_theme("dark")
    app.mainloop()