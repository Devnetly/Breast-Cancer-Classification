import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import torch
from threading import Thread
from feature_extraction import transform_wsis
from attention import Predict
import sv_ttk
import requests



class LoginPage(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Login")
        self.geometry("600x600")

        # Create labels and entry fields for username and password
        username_label = ttk.Label(self, text="Username:")
        username_label.pack(pady=10)
        self.username_entry = ttk.Entry(self)
        self.username_entry.pack()

        password_label = ttk.Label(self, text="Password:")
        password_label.pack(pady=10)
        self.password_entry = ttk.Entry(self, show="*")
        self.password_entry.pack()

        # Create a login button
        login_button = ttk.Button(self, text="Login", command=self.login)
        login_button.pack(pady=10)

    def login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()

        # Send a POST request to the login endpoint
        url = "http://127.0.0.1:8000/auth/login/"
        data = {"username": username, "password": password}
        response = requests.post(url, data=data)

        if response.status_code == 200:
            # If authentication is successful, store the access token and open the PatientListPage
            access_token = response.json().get("access")
            print(f"Access Token: {access_token}")
            self.destroy()
            patient_list_page = PatientListPage(access_token)
            sv_ttk.set_theme("dark")
            patient_list_page.mainloop()
        else:
            # Display an error message if authentication fails
            tk.messagebox.showerror("Login Error", "Invalid username or password")

import tkinter as tk
from tkinter import ttk

class PatientListPage(tk.Tk):
    def __init__(self, access_token):
        super().__init__()
        self.title("Patient List")
        self.geometry("600x600")
        self.access_token = access_token

        # Create a main frame
        main_frame = ttk.Frame(self)
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Send a GET request to retrieve the list of patients
        url = "http://127.0.0.1:8000/patients/"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            print(response.json())
            patients = response.json()

            # Create a treeview to display the patient list
            patient_treeview = ttk.Treeview(main_frame)
            patient_treeview.pack(pady=10, fill=tk.BOTH, expand=True)

            # Define columns
            patient_treeview["columns"] = ("id", "first_name", "last_name")
            patient_treeview.column("#0", width=0, stretch=tk.NO)  # Hide the first column
            patient_treeview.column("id", anchor=tk.W, width=100)
            patient_treeview.column("first_name", anchor=tk.W, width=150)
            patient_treeview.column("last_name", anchor=tk.W, width=150)

            # Create column headings
            patient_treeview.heading("#0", text="", anchor=tk.W)
            patient_treeview.heading("id", text="ID", anchor=tk.W)
            patient_treeview.heading("first_name", text="First Name", anchor=tk.W)
            patient_treeview.heading("last_name", text="Last Name", anchor=tk.W)

            # Insert data into the treeview
            for patient in patients:
                patient_treeview.insert("", tk.END, values=(patient["id"], patient["first_name"], patient["last_name"]))

            # Bind the selection event
            patient_treeview.bind("<<TreeviewSelect>>", self.on_patient_select)

        else:
            tk.messagebox.showerror("Error", "Failed to retrieve patient list")
            print(response.text)

    def on_patient_select(self, event):
        selected_item = event.widget.focus()
        if selected_item:
            patient_id = event.widget.item(selected_item)["values"][0]

            # Open the FileOpener with the selected patient ID
            self.destroy()
            app = FileOpener(patient_id, self.access_token)
            sv_ttk.set_theme("dark")
            app.mainloop()


class FileOpener(tk.Tk):
    def __init__(self, patient_id, access_token):
        super().__init__()
        self.title("test")
        self.geometry("600x600")

        print(f"Selected Patient ID: {patient_id}")
        print(f"Access Token: {access_token}")

        self.file_path_var = tk.StringVar()

        file_path_label = ttk.Label(self, text="WSI File Path:")
        file_path_label.pack(pady=10)

        file_path_entry = ttk.Entry(self, textvariable=self.file_path_var, width=30)
        file_path_entry.pack()

        browse_button = ttk.Button(self, text="Browse", command=self.browse_file)
        browse_button.pack(pady=10)

        open_button = ttk.Button(self, text="Extract", command=self.open_file)
        open_button.pack()

        self.progress_var = tk.StringVar()
        progress_label = ttk.Label(self, textvariable=self.progress_var)
        progress_label.pack(pady=10)

        self.prediction_var = tk.StringVar()
        prediction_label = ttk.Label(self, textvariable=self.prediction_var)
        prediction_label.pack(pady=10)

        predict_button = ttk.Button(self, text="Predict", command=self.predict)
        predict_button.pack()

        predict_button.config(state="disabled")

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("WSI Files", "*.svs")])
        self.file_path_var.set(file_path)

    def open_file(self):
        file_path = self.file_path_var.get()
        if file_path:
            predict_button = self.children["!button3"]  # Define the "predict_button" variable
            thread = Thread(target=transform_wsis, kwargs={
                'model': None,
                'source_path': file_path,
                'patch_size': 224,
                'destination_folder': './output',
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'prefetch_factor': 2,
                'app': self
            })
            thread.start()

            self.after(0, lambda: predict_button.config(state="normal"))

    def update_progress(self, message):
        def update():
            self.progress_var.set(message)
            self.update_idletasks()

        self.after(0, update)

    def update_prediction(self, message):
        def update():
            self.prediction_var.set(message)
            self.update_idletasks()

        self.after(0, update)

    def predict(self):
        # Get the path of the saved feature matrix
        path = "./output/output.pth"  # Replace with the actual path

        # Load the feature matrix
        tensor = torch.load(path)

        # Call the imported Predict function from attention.py
        prediction_result = Predict(tensor)

        # Print or display the prediction result
        print(f"Prediction Result: {prediction_result}")
        #example of output: Prediction Result: tensor([[0.2722, 0.4555, 0.2722]], grad_fn=<SoftmaxBackward0>)
        print(f"Bening: {prediction_result[0][0].item()}, Malignant: {prediction_result[0][1].item()}, Atypical: {prediction_result[0][2].item()}")
        #print the prediction in the app
        self.update_prediction(f"Bening: {prediction_result[0][0].item():.2f}, Malignant: {prediction_result[0][1].item():.2f}, Atypical: {prediction_result[0][2].item():.2f}")

if __name__ == "__main__":
    login_page = LoginPage()
    sv_ttk.set_theme("dark")
    login_page.mainloop()