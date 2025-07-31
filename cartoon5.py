import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, font
from PIL import Image, ImageTk

# --- Main Application Class ---
class GhibliStylerApp:
    def __init__(self, master):
        self.master = master
        self.master.title('Ghibli-Style Image Styler')
        self.master.geometry('1200x700')
        self.master.minsize(800, 600)
        self.master.configure(background='#2E2E2E')

        # --- Initialize variables ---
        self.original_image = None
        self.processed_image = None
        self.image_path = None
        
        # --- Define Fonts and Colors ---
        self.colors = {
            'bg': '#2E2E2E',
            'frame': '#3C3C3C',
            'text': '#FFFFFF',
            'button': '#555555',
            'button_fg': '#FFFFFF',
            'accent': '#007ACC'
        }
        self.header_font = font.Font(family='Helvetica', size=14, weight='bold')
        self.label_font = font.Font(family='Helvetica', size=10)
        self.button_font = font.Font(family='Helvetica', size=11, weight='bold')

        # --- Configure the main grid layout ---
        self.master.grid_rowconfigure(1, weight=1)
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)
        self.master.grid_columnconfigure(2, minsize=250, weight=0)

        # --- GUI Layout ---
        self._create_widgets()
        
        # --- Bind resize events ---
        self.original_canvas.bind("<Configure>", self.on_canvas_resize)
        self.stylized_canvas.bind("<Configure>", self.on_canvas_resize)

    def _create_widgets(self):
        """Creates and places all the widgets in the window."""
        top_frame = tk.Frame(self.master, bg=self.colors['bg'])
        top_frame.grid(row=0, column=0, columnspan=3, sticky='ew', pady=10, padx=10)

        self.upload_button = tk.Button(top_frame, text="Upload Image", command=self.upload_image, font=self.button_font, bg=self.colors['accent'], fg=self.colors['button_fg'], relief=tk.FLAT, padx=10)
        self.upload_button.pack(side=tk.LEFT, padx=(0, 10))

        self.save_button = tk.Button(top_frame, text="Save Image", command=self.save_image, font=self.button_font, bg=self.colors['button'], fg=self.colors['button_fg'], relief=tk.FLAT, state=tk.DISABLED, padx=10)
        self.save_button.pack(side=tk.LEFT)

        self.original_frame = self._create_image_frame("Original Image", 0)
        self.stylized_frame = self._create_image_frame("Stylized Image", 1)

        self.original_canvas = tk.Canvas(self.original_frame, bg='black', bd=0, highlightthickness=0)
        self.original_canvas.grid(row=1, column=0, sticky='nsew', padx=5, pady=(0,5))

        self.stylized_canvas = tk.Canvas(self.stylized_frame, bg='black', bd=0, highlightthickness=0)
        self.stylized_canvas.grid(row=1, column=0, sticky='nsew', padx=5, pady=(0,5))

        self._create_controls_panel()

    def _create_image_frame(self, text, col):
        """Helper to create the frames for holding images."""
        frame = tk.Frame(self.master, bg=self.colors['frame'], bd=2, relief=tk.SUNKEN)
        frame.grid(row=1, column=col, sticky='nsew', padx=(10, 5), pady=(0, 10))
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        label = tk.Label(frame, text=text, font=self.header_font, bg=self.colors['frame'], fg=self.colors['text'])
        label.grid(row=0, column=0, pady=5)
        return frame
        
    def _create_controls_panel(self):
        """Creates the right-side panel with all the sliders."""
        controls_frame = tk.Frame(self.master, bg=self.colors['frame'], bd=2, relief=tk.SUNKEN, width=250)
        controls_frame.grid(row=1, column=2, sticky='ns', padx=(5, 10), pady=(0, 10))
        controls_frame.pack_propagate(False)

        tk.Label(controls_frame, text="Style Controls", font=self.header_font, bg=self.colors['frame'], fg=self.colors['text']).pack(pady=10)

        self.palette_size = self._create_slider(controls_frame, "Palette Size (Colors)", 4, 32, 16)
        self.smoothing_level = self._create_slider(controls_frame, "Smoothing Level", 3, 15, 7, 2)
        self.saturation = self._create_slider(controls_frame, "Saturation Boost", 10, 25, 15, 1) # Scaled by 0.1
        
        reset_button = tk.Button(controls_frame, text="Reset to Defaults", command=self.reset_sliders, font=self.button_font, bg=self.colors['button'], fg=self.colors['button_fg'], relief=tk.FLAT)
        reset_button.pack(pady=20, padx=10, fill=tk.X)

    def _create_slider(self, parent, label, from_, to, default, resolution=1):
        """Helper to create a styled slider."""
        slider_frame = tk.Frame(parent, bg=self.colors['frame'])
        tk.Label(slider_frame, text=label, font=self.label_font, bg=self.colors['frame'], fg=self.colors['text']).pack()
        slider = tk.Scale(slider_frame, from_=from_, to=to, orient=tk.HORIZONTAL, resolution=resolution,
                          command=self.apply_style_effect, bg=self.colors['frame'], fg=self.colors['text'],
                          highlightbackground=self.colors['frame'], troughcolor='#555555', activebackground=self.colors['accent'])
        slider.set(default)
        slider.pack(fill=tk.X, padx=10)
        slider_frame.pack(pady=5, fill=tk.X)
        return slider

    def reset_sliders(self):
        """Resets all sliders to their default values."""
        self.palette_size.set(16)
        self.smoothing_level.set(7)
        self.saturation.set(15)
        self.apply_style_effect()

    def upload_image(self):
        """Opens a file dialog to select an image."""
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if not path: return
        self.image_path = path
        
        try:
            # FIX: Use np.fromfile and cv2.imdecode for robust path handling
            n = np.fromfile(path, np.uint8)
            self.original_image = cv2.imdecode(n, cv2.IMREAD_COLOR)
            
            if self.original_image is None:
                raise ValueError("Image file could not be read or is corrupted.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image from {path}\n\nDetails: {e}")
            return
        
        self.display_image(self.original_image, self.original_canvas)
        
        # --- FIX: Provide immediate user feedback before heavy processing ---
        self.show_processing_message()
        self.master.update_idletasks() # Force UI to update now
        
        # Defer the heavy processing to allow the UI to update
        self.master.after(50, self.apply_style_effect)
        
        self.save_button.config(state=tk.NORMAL, bg=self.colors['accent'])

    def show_processing_message(self):
        """Displays a 'Processing...' message on the stylized canvas."""
        self.stylized_canvas.delete("all")
        canvas_w = self.stylized_canvas.winfo_width()
        canvas_h = self.stylized_canvas.winfo_height()
        if canvas_w > 1 and canvas_h > 1:
            self.stylized_canvas.create_text(
                canvas_w/2, canvas_h/2, text="Processing...",
                font=self.header_font, fill=self.colors['text']
            )

    def apply_style_effect(self, event=None):
        """The core image processing function to create the Ghibli-like effect."""
        if self.original_image is None: return
        try:
            # --- FIX: Resize large images for performance ---
            max_dim = 800
            h, w, _ = self.original_image.shape
            if h > max_dim or w > max_dim:
                scale = max_dim / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                source_image = cv2.resize(self.original_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                source_image = self.original_image.copy()

            # --- Get Slider Values ---
            k = self.palette_size.get()
            smoothing = self.smoothing_level.get()
            saturation_scale = self.saturation.get() / 10.0

            # --- 1. Pre-smoothing ---
            d = smoothing * 2
            sigma = smoothing * 12
            smoothed_image = cv2.bilateralFilter(source_image, d=d, sigmaColor=sigma, sigmaSpace=sigma)

            # --- 2. Color Quantization using K-Means Clustering ---
            pixels = np.float32(smoothed_image.reshape(-1, 3))
            
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            centers = np.uint8(centers)
            
            quantized_image = centers[labels.flatten()]
            quantized_image = quantized_image.reshape(source_image.shape)

            # --- 3. Boost Saturation on the Painterly Image ---
            hsv_image = cv2.cvtColor(quantized_image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_image)
            s = cv2.multiply(s, saturation_scale)
            s = np.clip(s, 0, 255)
            hsv_image = cv2.merge([h, s, v])
            saturated_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
            
            self.processed_image = saturated_image

            self.display_image(self.processed_image, self.stylized_canvas)
        except Exception as e:
            messagebox.showerror("Processing Error", f"An error occurred during styling: {e}")

    def on_canvas_resize(self, event):
        """Redraw images when the canvas is resized."""
        if event.widget == self.original_canvas and self.original_image is not None:
            self.display_image(self.original_image, self.original_canvas)
        elif event.widget == self.stylized_canvas and self.processed_image is not None:
            self.display_image(self.processed_image, self.stylized_canvas)

    def display_image(self, img_cv, canvas):
        """Robustly resizes and displays a cv2 image on a Tkinter canvas."""
        try:
            canvas_w = canvas.winfo_width()
            canvas_h = canvas.winfo_height()

            if canvas_w <= 1 or canvas_h <= 1:
                self.master.after(50, lambda: self.display_image(img_cv, canvas))
                return
                
            h, w, _ = img_cv.shape
            scale = min(canvas_w / w, canvas_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            if new_w <= 0 or new_h <= 0: return

            resized_img = cv2.resize(img_cv, (new_w, new_h), interpolation=cv2.INTER_AREA)
            img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            
            canvas.delete("all")
            canvas.create_image(canvas_w/2, canvas_h/2, anchor=tk.CENTER, image=img_tk)
            canvas.image = img_tk
        except Exception as e:
            print(f"Error in display_image: {e}")

    def save_image(self):
        """Saves the processed stylized image."""
        if self.processed_image is None or self.image_path is None:
            messagebox.showerror("Error", "No stylized image to save.")
            return

        original_dir = os.path.dirname(self.image_path)
        original_filename, _ = os.path.splitext(os.path.basename(self.image_path))
        
        save_path = filedialog.asksaveasfilename(
            initialdir=original_dir,
            initialfile=f"{original_filename}_ghibli_style.png",
            defaultextension=".png",
            filetypes=[("PNG file", "*.png"), ("JPG file", "*.jpg"), ("All files", "*.*")]
        )

        if save_path:
            try:
                cv2.imwrite(save_path, self.processed_image)
                messagebox.showinfo("Success", f"Image saved successfully at:\n{save_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save image: {e}")

# --- Main execution block ---
if __name__ == "__main__":
    # Before you run, make sure you have the necessary libraries installed:
    # pip install opencv-python
    # pip install Pillow
    # pip install numpy
    root = tk.Tk()
    app = GhibliStylerApp(root)
    root.mainloop()
