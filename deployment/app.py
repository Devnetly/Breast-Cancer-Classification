import os
import dotenv
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import torch
from threading import Thread
from feature_extraction import transform_wsis
from hipt import Predict
import sv_ttk
import requests
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

try:
    OPENSLIDE_PATH = dotenv.get_key(dotenv.find_dotenv(), "OPENSLIDE_PATH")
except Exception as e:
    print("Error setting OpenSlide path:", str(e))


if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
        from openslide import open_slide
        from openslide.deepzoom import DeepZoomGenerator
else:
    import openslide
    from openslide import open_slide
    from openslide.deepzoom import DeepZoomGenerator



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
            user_id = response.json().get("user_id")
            print(f"Access Token: {access_token}")
            self.destroy()
            patient_list_page = PatientListPage(access_token, user_id)
            sv_ttk.set_theme("dark")
            patient_list_page.mainloop()
        else:
            # Display an error message if authentication fails
            tk.messagebox.showerror("Login Error", "Invalid username or password")

class PatientListPage(tk.Tk):
    def __init__(self, access_token, user_id):
        super().__init__()
        self.title("Patient List")
        self.geometry("600x600")
        #self.attributes('-fullscreen', True)
        self.access_token = access_token
        self.user_id = user_id

        # Create a main frame
        main_frame = ttk.Frame(self)
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Create a search frame
        search_frame = ttk.Frame(main_frame)
        search_frame.pack(pady=10, fill=tk.X)

        # Search entry field
        self.search_entry = ttk.Entry(search_frame)
        self.search_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Search button
        search_button = ttk.Button(search_frame, text="Search", command=self.search_patients)
        search_button.pack(side=tk.LEFT, padx=5)

        # Create a treeview to display the patient list
        self.patient_treeview = ttk.Treeview(main_frame)
        self.patient_treeview.pack(pady=10, fill=tk.BOTH, expand=True)

        # Define columns
        self.patient_treeview["columns"] = ("id", "first_name", "last_name")
        self.patient_treeview.column("#0", width=0, stretch=tk.NO)  # Hide the first column
        self.patient_treeview.column("id", anchor=tk.W, width=100)
        self.patient_treeview.column("first_name", anchor=tk.W, width=150)
        self.patient_treeview.column("last_name", anchor=tk.W, width=150)

        # Create column headings
        self.patient_treeview.heading("#0", text="", anchor=tk.W)
        self.patient_treeview.heading("id", text="ID", anchor=tk.W)
        self.patient_treeview.heading("first_name", text="First Name", anchor=tk.W)
        self.patient_treeview.heading("last_name", text="Last Name", anchor=tk.W)

        # Fetch and populate the patient list
        self.fetch_patients()

        # Bind the selection event
        self.patient_treeview.bind("<<TreeviewSelect>>", self.on_patient_select)

    def fetch_patients(self):
        url = "http://127.0.0.1:8000/cac/medics/" + str(self.user_id) + "/"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            patients = response.json()
            self.populate_treeview(patients)
        else:
            tk.messagebox.showerror("Error", "Failed to retrieve patient list")
            print(response.text)

    def populate_treeview(self, patients):
        # Clear the treeview
        self.patient_treeview.delete(*self.patient_treeview.get_children())

        # Insert data into the treeview
        for patient in patients:
            self.patient_treeview.insert("", tk.END, values=(patient["id"], patient["first_name"], patient["last_name"]))

    def search_patients(self):
        search_term = self.search_entry.get().lower()
        url = "http://127.0.0.1:8000/cac/medics/" + str(self.user_id) + "/"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            patients = response.json()
            filtered_patients = [patient for patient in patients if search_term in (str(patient["id"]) + patient["first_name"].lower() + patient["last_name"].lower())]
            self.populate_treeview(filtered_patients)
        else:
            tk.messagebox.showerror("Error", "Failed to retrieve patient list")
            print(response.text)

    def on_patient_select(self, event):
        selected_item = event.widget.focus()
        if selected_item:
            patient_id = event.widget.item(selected_item)["values"][0]

            # Open the FileOpener with the selected patient ID
            self.destroy()
            app = HistologiesListPage(self.access_token, patient_id)
            sv_ttk.set_theme("dark")
            app.mainloop()

class HistologiesListPage(tk.Tk):
    def __init__(self, access_token, patient_id):
        super().__init__()
        self.title("Histologies List")
        self.geometry("600x600")
        #self.attributes('-fullscreen', True)
        self.access_token = access_token
        self.patient_id = patient_id

        # Create a main frame
        main_frame = ttk.Frame(self)
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Create a search frame
        search_frame = ttk.Frame(main_frame)
        search_frame.pack(pady=10, fill=tk.X)

        # Search entry field
        self.search_entry = ttk.Entry(search_frame)
        self.search_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Create a treeview to display the histologies list
        self.histologies_treeview = ttk.Treeview(main_frame)
        self.histologies_treeview.pack(pady=10, fill=tk.BOTH, expand=True)

        # Define columns
        self.histologies_treeview["columns"] = ("id", "zone", "yes_no", "rank", "cancer_type", "molecular_profile")
        self.histologies_treeview.column("#0", width=0, stretch=tk.NO)  # Hide the first column
        self.histologies_treeview.column("zone", anchor=tk.W, width=150)
        self.histologies_treeview.column("yes_no", anchor=tk.W, width=150)
        self.histologies_treeview.column("rank", anchor=tk.W, width=150)
        self.histologies_treeview.column("cancer_type", anchor=tk.W, width=150)
        self.histologies_treeview.column("molecular_profile", anchor=tk.W, width=150)

        # Create column headings
        self.histologies_treeview.heading("#0", text="", anchor=tk.W)
        self.histologies_treeview.heading("id", text="ID", anchor=tk.W)
        self.histologies_treeview.heading("zone", text="Zone", anchor=tk.W)
        self.histologies_treeview.heading("yes_no", text="Yes/No", anchor=tk.W)
        self.histologies_treeview.heading("rank", text="Rank", anchor=tk.W)
        self.histologies_treeview.heading("cancer_type", text="Type", anchor=tk.W)
        self.histologies_treeview.heading("molecular_profile", text="Profile", anchor=tk.W)

        # Fetch and populate the histologies list
        self.fetch_histologies()

        # Bind the selection event
        self.histologies_treeview.bind("<<TreeviewSelect>>", self.on_histology_select)

    def fetch_histologies(self):
        url = "http://localhost:8000/cac/"+str(self.patient_id)+"/histologies/"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            histologies = response.json()
            self.populate_treeview(histologies)
        else:
            tk.messagebox.showerror("Error", "Failed to retrieve histologies list")
            print(response.text)

    def populate_treeview(self, histologies):
        # Clear the treeview
        self.histologies_treeview.delete(*self.histologies_treeview.get_children())

        # Insert data into the treeview
        for histology in histologies:
            self.histologies_treeview.insert("", tk.END, values=(histology["id"], histology["zone"], histology["yes_no"], histology["rank"], histology["cancer_type"], histology["molecular_profile"]))

    def on_histology_select(self, event):
        selected_item = event.widget.focus()
        if selected_item:
            histology_id = event.widget.item(selected_item)["values"][0]

            # Open the FileOpener with the selected patient ID
            self.destroy()
            app = FileOpener(histology_id, self.access_token)
            sv_ttk.set_theme("dark")
            app.mainloop()


class FileOpener(tk.Tk):
    def __init__(self, histology_id, access_token):
        super().__init__()
        self.title("Inference App")
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry(f"{screen_width}x{screen_height}")

        print(f"Selected Histology ID: {histology_id}")
        print(f"Access Token: {access_token}")
        self.histology_id = histology_id
        self.access_token = access_token

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

        self.canvas = tk.Canvas(self, width=600, height=450)  # Adjust the width and height as needed
        self.canvas.pack()

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("WSI Files", "*.svs")])
        self.file_path_var.set(file_path)

        def slide_to_scaled_pil_image(slide, SCALE_FACTOR=64):
            level = slide.get_best_level_for_downsample(SCALE_FACTOR)
            new_w, new_h = slide.level_dimensions[level]
            whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
            whole_slide_image = whole_slide_image.convert("RGB")
            img = whole_slide_image.resize((new_w, new_h), Image.BILINEAR)
            return img, new_w, new_h

        if file_path:
            # Display the WSI
            slide = openslide.OpenSlide(file_path)
            img, new_w, new_h = slide_to_scaled_pil_image(slide, SCALE_FACTOR=128)
            print(new_w, new_h)

            # Convert the PIL image to a PhotoImage and display it on the canvas
            photo_image = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo_image)
            self.canvas.image = photo_image  # Keep a reference to prevent garbage collection

    def open_file(self):
        file_path = self.file_path_var.get()
        if file_path:
            predict_button = self.children["!button3"]  # Define the "predict_button" variable
            thread = Thread(target=transform_wsis, kwargs={
                'model': None,
                'source_path': file_path,
                'patch_size': 4096,
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
        print(f"Atypical: {prediction_result[0][0].item()}, Benign: {prediction_result[0][1].item()}, Malignant: {prediction_result[0][2].item()}")
        #print the prediction in the app
        self.update_prediction(f"Bening: {prediction_result[0][0].item():.2f}, Malignant: {prediction_result[0][1].item():.2f}, Atypical: {prediction_result[0][2].item():.2f}")
        #post the prediction to the API
        url = "http://127.0.0.1:8000/cac/predictions/"+str(self.histology_id)+"/"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        #take two decimals
        prediction_result = [round(i.item(), 2) for i in prediction_result[0]]
        data = {"atypical": prediction_result[0], "benign": prediction_result[1], "malignant": prediction_result[2], "histology": self.histology_id}
        response = requests.post(url, headers=headers, data=data)
        if response.status_code == 201:
            print("Prediction posted successfully")
        else:
            print("Failed to post prediction")
            print(response.text)


if __name__ == "__main__":
    login_page = LoginPage()
    sv_ttk.set_theme("dark")
    login_page.mainloop()