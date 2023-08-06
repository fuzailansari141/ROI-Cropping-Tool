import os
import cv2
import tkinter
import warnings
import numpy as np
import pandas as pd
import tkinter as tk
import customtkinter
from PIL import Image
import tkinter.messagebox
warnings.filterwarnings('ignore')
import tkinter.messagebox as messagebox
from tkinter import filedialog
customtkinter.set_appearance_mode("Light")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue")
#%%
class App(customtkinter.CTk):
    width=1100
    height=580
    def __init__(self):
        super().__init__()
        self.characters=[]
        # configure window
        self.title("ROI Image Processing Tool")
        self.geometry(f"{self.width}x{self.height}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        
        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=5, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(5, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="MENU",font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame,text="Load Image", command=self.load_image)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.combobox_1 = customtkinter.CTkComboBox(self.sidebar_frame,values=self.characters)
        self.combobox_1.grid(row=2, column=0, padx=20, pady=(10, 10))
        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame,text="ROI Cropping", command=self.process_with_characteristic)
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)
        self.sidebar_button_4 = customtkinter.CTkButton(self.sidebar_frame,text="Combine Csv", command=self.combine)
        self.sidebar_button_4.grid(row=4, column=0, padx=20, pady=10)
        
        self.toplevel_window=None
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", font=customtkinter.CTkFont(weight="bold"),anchor="w")
        self.appearance_mode_label.grid(row=6, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=7, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:",font=customtkinter.CTkFont(weight="bold"),anchor="w")
        self.scaling_label.grid(row=8, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=9, column=0, padx=20, pady=(10, 20))     
        
        # load and create background image
        self.bg_image = customtkinter.CTkImage(Image.open(r"C:\Users\Fuzail Ansari\Downloads\paper-style-white-monochrome-background\5605708.jpg"),
                                                size=(1400,800))
        self.bg_image_label = customtkinter.CTkLabel(self,text="",image= self.bg_image)
        self.bg_image_label.grid(row=0, column=1,rowspan=4)
        
        # create home frame
        self.home_frame = customtkinter.CTkFrame(self, corner_radius=10,width=500, height=300)
        self.home_frame.grid(row=0, column=1, padx=(18, 0), pady=(18, 0))
        
        # create textbox
        self.textbox = customtkinter.CTkTextbox(self, corner_radius=5,width=600,height=200)
        self.textbox.grid(row=1, column=1, padx=(18, 0), pady=(18, 0))
        
        self.textbox.configure(state="disabled")
#%%              
    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)       
#%%
    def redirect_print(self,text):
        self.textbox.configure(state="normal")
        self.textbox.insert(tk.END, text + "\n")
        self.textbox.see(tk.END)
        self.textbox.configure(state="disabled")
#%%
    # Function to load the image
    def load_image(self):
        global file_path, img, folder_path, folder_selected,process
        self.redirect_print("Loading Image....")
        app.update()
        file_path = tk.filedialog.askopenfilename(
            title="Select Image",
            filetypes=(("Image files", "*.jpg;*.jpeg;*.png;*.TIF;*.tif"), ("All files", "*.*"))
        )
        if file_path:
            # Open the selected image using PIL
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            self.my_image = customtkinter.CTkImage(light_image=Image.open(file_path),
                                      dark_image=Image.open(file_path),
                                      size=(400, 250))
            self.image_label = customtkinter.CTkLabel(self.home_frame, text="", image=self.my_image)
            self.image_label.grid(row=0, column=0, padx=20, pady=10)        
            self.redirect_print("Image Loaded")
            app.update()
            
            file_name = os.path.basename(file_path)
            folder_selected=False
            if not folder_selected:
                self.redirect_print("selecting folder path...")
                app.update()
                # Prompt the user to select the output directory
                root = tk.Tk()
                root.withdraw()
                output_dir = filedialog.askdirectory(title="Select Path to Save Files")
                
                if not output_dir:
                    return  # User canceled the folder selection
                
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                # Create a new folder within the selected output directory
                folder_name = file_name.split(".")[0]
                folder_path = os.path.join(output_dir, folder_name)
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
                
                folder_selected = True
            self.redirect_print("Folder path Selected")
            app.update()
            self.redirect_print("Saving Original image Csv...")
            app.update()
            # Save original image as CSV in the new folder
            csv_path = os.path.join(folder_path,"original_image.csv")
            
            # Save all bands as CSV
            np.savetxt(csv_path, img.reshape(-1, img.shape[2]), delimiter=',', fmt='%d', header=','.join([f'Band {i}' for i in range(1, img.shape[2] + 1)]),comments='')
            self.redirect_print("Original image Csv Saved")
            app.update()

#%%
    def process(self,img,quality):
        global bl_img, coords, rois
        height, width,channels= img.shape
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', width, height)
        cv2.imshow('image', img)

        coords = []
        rois = []
        bl_img = None

        def on_mouse_click(event, x, y, flags, params):
            global bl_img, coords, rois
            if event == cv2.EVENT_LBUTTONDOWN:
                coords.append((x, y))
                if len(coords) > 1:
                    # Draw a line between the last two clicked points for reference
                    cv2.line(img, coords[-2], coords[-1], (0, 0, 0), 2)
                    cv2.imshow('image', img)
            if event == cv2.EVENT_RBUTTONDOWN:
                if len(coords) >= 3:  # Check if ROI has at least 3 points
                    rois.append(coords)
                    mask = np.zeros_like(img)
                    cv2.fillPoly(mask, [np.array(coords)], (255,) * channels)
                    roi_img = cv2.bitwise_and(img, mask)
                    if np.max(roi_img) == 0:
                        messagebox.showerror("Error", "ROI Cannot be Colinear")
                    else:
                        blank_img = np.zeros_like(img)
                        for roi in rois:
                            mask = np.zeros_like(img)
                            cv2.fillPoly(mask, [np.array(roi)], (255,) * channels)
                            roi_img = cv2.bitwise_and(img, mask)
                            blank_img = cv2.add(blank_img, roi_img)
                        bl_img = blank_img
                        coords = []
                else:
                    messagebox.showerror("Error", "ROI must have at least 3 points")

        cv2.setMouseCallback('image', on_mouse_click)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        ans = messagebox.askquestion("Regenerate ROI", "Do you want to regenerate the ROI?")
        if ans == 'yes':
            img = cv2.imread(file_path)
            self.process(img,quality)
        else:
            if bl_img is not None:
                roi_path = os.path.join(folder_path, f"{quality}_roi.tiff")
                cv2.imwrite(roi_path, bl_img)
                cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
                cv2.imshow('Result', bl_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                self.redirect_print("Saving ROI image Csv....")
                app.update()
                # Save original image as CSV in the new folder
                csv_path = os.path.join(folder_path,f"{quality}_image.csv")
                
                # Save all bands as CSV
                np.savetxt(csv_path, bl_img.reshape(-1, bl_img.shape[2]), delimiter=',', fmt='%d', header=','.join([f'Band {i}' for i in range(1, bl_img.shape[2] + 1)]),comments='')
                
                df = pd.read_csv(csv_path)
                
                w = df.sum(axis=1)

                df['Characteristics'] = ['null' if i == 0 else quality for i in w]
                
                df.to_csv(csv_path,index=False)
            else:
                messagebox.showerror("Error", "No ROI selected.")
        self.redirect_print("Cropping Completed & File Saved ")
        app.update()
#%%   
    def combine(self):
        self.redirect_print("Combining csv...")
        app.update()
        data_frames = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv') and  file_name.split(".")[0] != 'original_image':
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path)
                data_frames.append(df)

        # Combine data from all CSV files
        combined_df = data_frames[0].copy()

        for i in range(len(combined_df)):
            if pd.isna(combined_df.at[i, 'Characteristics']):
                for j in range(1, len(data_frames)):
                    value = data_frames[j].at[i, 'Characteristics']
                    if not pd.isna(value):
                        combined_df.iloc[i] = data_frames[j].iloc[i]
                        break
        csv_path = os.path.join(folder_path,'combined.csv')
        # Save the combined DataFrame as a new CSV file
        combined_df.to_csv(csv_path, index=False)
        self.redirect_print("csv combined.")
        app.update()
#%%
    def process_with_characteristic(self):
        selected_characteristic = self.combobox_1.get()
        if selected_characteristic:
            global img
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            quality = selected_characteristic
            self.process(img, quality)
            self.update_character_combobox(quality)
        else:
            messagebox.showerror("Error", "Please select a characteristic.")
#%%
    def update_character_combobox(self,characters):
        self.characters.append(characters)
        self.characters=set(self.characters)
        self.characters=list(self.characters)
        self.combobox_1 = customtkinter.CTkComboBox(self.sidebar_frame,values=self.characters)
        self.combobox_1.grid(row=2, column=0, padx=20, pady=(10, 10))
#%%
if __name__ == "__main__":
    app = App()
    app.mainloop()
#%%