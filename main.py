import tkinter as tk
from tkinter import Menu, filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageEnhance
from tkinter import Label, Entry, Scale, Button
from scipy.ndimage import gaussian_filter
import cv2
import numpy as np
import scipy.ndimage as ndimage
from skimage.util import random_noise
from scipy.fftpack import dct, idct
import math

root = tk.Tk()
root.title("Photoshop App")
root.geometry("1250x850")

original_image_pil = None
current_image_pil = None
edited_image = None
undo_stack = []
redo_stack = []
start_x = None
start_y = None
rect = None

def icon(icon_image):
    img = Image.open(icon_image)
    img = img.resize((24, 24), Image.Resampling.BICUBIC)
    return ImageTk.PhotoImage(img)

def create_menu_bar():
    # Tạo menu bar
    menu_bar = Menu(root, bg="#255788", fg="white", activebackground="#3ea7ec", activeforeground="white")
    #fg là màu chữ
    # Create "File" menu
    file_menu = Menu(menu_bar, tearoff=0, bg="#255788", fg="white", activebackground="#3ea7ec", activeforeground="white")
    open_icon = icon("load.png")
    save_icon = icon("save.png")
    file_menu.add_command(label="Open", command = open_image, image = open_icon, compound = "left")
    file_menu.add_command(label="Save", command = save_image, image = save_icon, compound = "left")
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit)
    menu_bar.add_cascade(label="File", menu=file_menu)

    # Create "Edit" menu
    edit_menu = Menu(menu_bar, tearoff=0, bg="#255788", fg="white", activebackground="#3ea7ec", activeforeground="white")
    crop_icon = icon("crop.png")
    compression_icon = icon("compression.png")
    resize_icon = icon("resize.png")
    transform_icon = icon("transform.png")
    edit_menu.add_command(label="Crop", command = crop_image, image = crop_icon, compound = "left")
    edit_menu.add_command(label="Compression", command = compression_image, image = compression_icon, compound = "left")
    edit_menu.add_command(label="Resize", command=resize_image, image=resize_icon, compound="left")
    edit_menu.add_command(label="Transform", command = transform_image, image = transform_icon, compound = "left")
    menu_bar.add_cascade(label="Edit", menu=edit_menu)

    # Create "Image" menu
    image_menu = Menu(menu_bar, tearoff=0, bg="#255788", fg="white", activebackground="#3ea7ec", activeforeground="white")
    gray_icon = icon("gray.png")
    binary_icon = icon("binary.png")
    adjustment_icon = icon("adjustment.png")
    image_menu.add_command(label="Gray Scale",command = gray_scale_image, image = gray_icon, compound = "left")
    image_menu.add_command(label="Binary", command = binary_image, image = binary_icon, compound = "left")
    image_menu.add_command(label="Adjustment", command = adjustment_image, image = adjustment_icon, compound = "left")
    menu_bar.add_cascade(label="Image", menu=image_menu)

# Create "Filter" menu
    filter_menu = Menu(menu_bar, tearoff=0, bg="#255788", fg="white", activebackground="#3ea7ec", activeforeground="white")
    edge_icon = icon("edgedetection.png")
    filter_icon = icon("filter.png")
    noise_icon = icon("noise.png")
    filter_menu.add_command(label="Edge Detection", command = edge_detection, image = edge_icon, compound = "left")
    filter_menu.add_command(label="Noise", command = add_noise, image = noise_icon, compound = "left")
    filter_menu.add_command(label="Filter", command = filter_image, image = filter_icon, compound = "left")
    menu_bar.add_cascade(label="Filter", menu=filter_menu)
    
    # Create "View" menu
    view_menu = Menu(menu_bar, tearoff=0, bg="#255788", fg="white", activebackground="#3ea7ec", activeforeground="white")
    zoomin_icon = icon("zoom-in.png")
    zoomout_icon = icon("zoom-out.png")
    fit_icon = icon("fit-screen.png")
    view_menu.add_command(label="Zoom In", command = zoom_in, image = zoomin_icon, compound = "left")
    view_menu.add_command(label="Zoom Out", command = zoom_out, image = zoomout_icon, compound = "left")
    view_menu.add_command(label="Fit on the screen", command = fit_on_screen, image = fit_icon, compound = "left")
    menu_bar.add_cascade(label="View", menu=view_menu)

    root.config(menu=menu_bar)

    root.image_refs = [open_icon, save_icon, crop_icon, compression_icon, resize_icon, transform_icon,
                       gray_icon, binary_icon, adjustment_icon, edge_icon, filter_icon,
                       noise_icon, zoomin_icon, zoomout_icon, fit_icon]

#***************CẮT ẢNH**************
def crop_image():
    global edit_canvas
    if current_image_pil is None:
        messagebox.showwarning("Warning", "Please open an image first!")
        return
    edit_canvas.bind("<ButtonPress-1>", on_button_press)
    edit_canvas.bind("<B1-Motion>", on_mouse_drag)
    edit_canvas.bind("<ButtonRelease-1>", on_button_release)
def on_button_press(event):
    global start_x, start_y, rect
    start_x = event.x
    start_y = event.y
    if rect:
        edit_canvas.delete(rect)
    rect = edit_canvas.create_rectangle(start_x, start_y, start_x, start_y, outline="red")

def on_mouse_drag(event):
    global rect
    edit_canvas.coords(rect, start_x, start_y, event.x, event.y)

def on_button_release(event):
    global start_x, start_y, rect, current_image_pil, display_width, display_height
    end_x, end_y = event.x, event.y
    if None not in (start_x, start_y, end_x, end_y) and current_image_pil is not None:
        # Ensure coordinates are in the correct order
        x1 = min(start_x, end_x)
        y1 = min(start_y, end_y)
        x2 = max(start_x, end_x)
        y2 = max(start_y, end_y)

        # Calculate the scaling factors based on the displayed image dimensions
        image_width, image_height = current_image_pil.size
        scale_x = image_width / display_width
        scale_y = image_height / display_height

        # Calculate canvas offsets to center the image if necessary
        canvas_width = edit_canvas.winfo_width()
        canvas_height = edit_canvas.winfo_height()
        x_offset = max((canvas_width - display_width) // 2, 0)
        y_offset = max((canvas_height - display_height) // 2, 0)

        # Adjust coordinates for the offset
        x1 = max(x1 - x_offset, 0)
        y1 = max(y1 - y_offset, 0)
        x2 = max(x2 - x_offset, 0)
        y2 = max(y2 - y_offset, 0)

        # Scale coordinates to match the original image size
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        # Ensure coordinates are within the image bounds
        x1 = max(min(x1, image_width), 0)
        y1 = max(min(y1, image_height), 0)
        x2 = max(min(x2, image_width), 0)
        y2 = max(min(y2, image_height), 0)

        # Crop the image if the rectangular region is valid
        if x1 != x2 and y1 != y2:
            push_to_undo_stack()
            cropped_image = current_image_pil.crop((x1, y1, x2, y2))
            current_image_pil = cropped_image
            display_image_in_edit_canvas(current_image_pil)
            display_image_info(current_image_pil)
    # Unbind the events to stop cropping
    edit_canvas.unbind("<ButtonPress-1>")
    edit_canvas.unbind("<B1-Motion>")
    edit_canvas.unbind("<ButtonRelease-1>")

    # Remove the rectangle
    if rect:
        edit_canvas.delete(rect)
#****************NÉN ẢNH***************
def compression_image():
    global current_image_pil, info_frame
    push_to_undo_stack()
    def apply_compression(compression_frame):
        global current_image_pil
        quality = int(quality_entry.get())

        # Chuyển đổi ảnh sang YCbCr
        ycbcr_image = current_image_pil.convert("YCbCr")

        # Chia ảnh thành các khối 8x8 và nén từng khối
        compressed_image = compress_ycbcr_blocks(ycbcr_image, quality)

        current_image_pil = compressed_image  # Cập nhật ảnh hiện tại
        display_image_in_edit_canvas(current_image_pil)
        display_image_info(current_image_pil)
        close_compression_frame(compression_frame)

    def compress_ycbcr_blocks(image, quality):
        y, cb, cr = image.split()  # Tách kênh YCbCr
        y_np = np.array(y, dtype=np.float32) # Chuyển đổi kênh Y sang mảng NumPy
        cb_np = np.array(cb) # Mảng cho kênh Cb
        cr_np = np.array(cr) # Mảng cho kênh Cr
    

    # DCT và lượng tử hóa cho từng khối 8x8
        rows, cols = y_np.shape
        pad_rows = 8 - rows % 8 if rows % 8 != 0 else 0
        pad_cols = 8 - cols % 8 if cols % 8 != 0 else 0
        y_np = np.pad(y_np, ((0, pad_rows), (0, pad_cols)), mode='constant')
        for i in range(0, rows, 8):
            for j in range(0, cols, 8):
                block = y_np[i:i+8, j:j+8]

                # DCT 2D bằng scipy.fftpack.dct
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

                # Lượng tử hóa
                threshold = quality * np.abs(dct_block).max() / 100
                dct_block[np.abs(dct_block) < threshold] = 0

                # IDCT 2D bằng scipy.fftpack.idct
                y_np[i:i+8, j:j+8] = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
         # Cắt bỏ padding sau khi IDCT
        y_np = y_np[:rows, :cols]
        y_compressed = Image.fromarray(np.uint8(y_np)) # Chuyển mảng NumPy thành ảnh PIL

    # Kết hợp lại ảnh
        compressed_image = Image.merge("YCbCr", (y_compressed, cb, cr)).convert("RGB")
        return compressed_image


    def close_compression_frame(compression_frame):  # Nhận compression_frame làm đối số
        compression_frame.destroy()

    # Tạo khung nén
    compression_frame = ttk.LabelFrame(info_frame, text="Compression", padding=10)
    compression_frame.pack(fill="x", expand=True, pady=5)

    # Nhập chất lượng
    Label(compression_frame, text="Quality (1-100):").grid(row=0, column=0, padx=5, pady=5)
    quality_entry = Entry(compression_frame, width=5)
    quality_entry.grid(row=0, column=1, padx=5, pady=5)
    quality_entry.insert(0, "80")

    # Nút Apply
    apply_button = Button(compression_frame, text="OK", command=lambda: apply_compression(compression_frame), font=("Arial", 12, "bold"), bg = "green", fg = "white")  # Truyền compression_frame
    apply_button.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

#***************Resize image*************
def resize_image():
    global current_image_pil, edited_image, resize_width_entry, resize_height_entry, resize_slider, apply_button, reset_button, resize_controls_frame
    push_to_undo_stack()
    if not hasattr(root, 'resize_width_entry'):
        create_resize_controls()
def create_resize_controls():
    global resize_width_entry, resize_height_entry, resize_slider, apply_button, reset_button, cancel_button
    global resize_controls_frame  # Khung riêng cho resize controls
    
    # Tạo khung con bên trong info_frame
    resize_controls_frame = tk.Frame(info_frame, relief="groove", borderwidth=2)
    resize_controls_frame.pack(fill="x", padx=5, pady=5)

    # Nhập width
    Label(resize_controls_frame, text="Width:").pack(fill="x", padx=5, pady=2)
    resize_width_entry = Entry(resize_controls_frame)
    resize_width_entry.pack(fill="x", padx=5, pady=2)

    # Nhập Height
    Label(resize_controls_frame, text="Height:").pack(fill="x", padx=5, pady=2)
    resize_height_entry = Entry(resize_controls_frame)
    resize_height_entry.pack(fill="x", padx=5, pady=2)
    tk.Frame(resize_controls_frame, height=10).pack()
    # Buttons
    button_frame = tk.Frame(resize_controls_frame)
    button_frame.pack(fill="x")
    
    apply_button = Button(button_frame, text="OK", command=apply_resize, font=("Arial", 12, "bold"), bg = "green", fg = "white")
    apply_button.pack(side=tk.LEFT, padx=5)
    
    reset_button = Button(button_frame, text="Reset", command=reset_size, font=("Arial", 12, "bold"), bg = "red", fg = "white")
    reset_button.pack(side=tk.LEFT, padx=5)

    cancel_button = Button(button_frame, text="Cancel", command=close_resize_controls, font=("Arial", 12, "bold"), bg = "gray", fg = "white")
    cancel_button.pack(side=tk.LEFT, padx=5)

    # Khởi tạo với hình ảnh kích thước hiện tại
    if current_image_pil:
        resize_width_entry.insert(0, str(current_image_pil.size[0]))
        resize_height_entry.insert(0, str(current_image_pil.size[1]))

def apply_resize():
    new_width = int(resize_width_entry.get())
    new_height = int(resize_height_entry.get())
    if new_width <= 0 or new_height <= 0:
        raise ValueError("Width and height must be positive.")
    global current_image_pil
    current_image_pil = current_image_pil.resize((new_width, new_height), Image.Resampling.BICUBIC)
    display_image_in_edit_canvas(current_image_pil)
    display_image_info(current_image_pil)

def reset_size():
    if original_image_pil:
        resize_width_entry.delete(0, tk.END)
        resize_width_entry.insert(0, str(original_image_pil.size[0]))
        resize_height_entry.delete(0, tk.END)
        resize_height_entry.insert(0, str(original_image_pil.size[1]))
def close_resize_controls():
    global resize_controls_frame
    if resize_controls_frame is not None:
        resize_controls_frame.destroy()  #hủy khung điều khiển resize
        resize_controls_frame = None
def transform_image():
    global current_image_pil, edited_image, info_frame
    push_to_undo_stack()
    if current_image_pil is None:
        messagebox.showwarning("Warning", "Please open an image first!")
        return
    def rotate90():
        global current_image_pil
        current_image_pil = current_image_pil.rotate(90, expand=True)
        display_image_in_edit_canvas(current_image_pil)
        display_image_info(current_image_pil)
    def rotate180():
        global current_image_pil
        current_image_pil = current_image_pil.rotate(180, expand=True)
        display_image_in_edit_canvas(current_image_pil)
        display_image_info(current_image_pil)
    def flip_horizontally():
        global current_image_pil
        current_image_pil = current_image_pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        display_image_in_edit_canvas(current_image_pil)
        display_image_info(current_image_pil)
    def flip_vertically():
        global current_image_pil
        current_image_pil = current_image_pil.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        display_image_in_edit_canvas(current_image_pil)
        display_image_info(current_image_pil)
    def close_transform():
        transform_frame.pack_forget()  # Remove the frame
    transform_frame = ttk.LabelFrame(info_frame, text="Transform", padding=10)  # Use a LabelFrame
    transform_frame.pack(fill="x", expand=True, pady=5)

    buttons_frame = ttk.Frame(transform_frame)
    buttons_frame.pack(padx=10, pady=5)

    ttk.Button(buttons_frame, text="Rotate 90°", command=rotate90).pack(fill="x", pady=5)
    ttk.Button(buttons_frame, text="Rotate 180°", command=rotate180).pack(fill="x", pady=5)
    ttk.Button(buttons_frame, text="Flip Horizontally", command=flip_horizontally).pack(fill="x", pady=5)
    ttk.Button(buttons_frame, text="Flip Vertically", command=flip_vertically).pack(fill="x", pady=5)
    ttk.Button(buttons_frame, text="OK", command=close_transform).pack(fill="x", pady=5)
def gray_scale_image():
    global current_image_pil, edited_image
    push_to_undo_stack()
    img = current_image_pil.convert("L")
    current_image_pil = img  
    display_image_in_edit_canvas(current_image_pil)
    display_image_info(current_image_pil)
def binary_image():
    global current_image_pil, edited_image
    push_to_undo_stack()
    gray_image = current_image_pil.convert("L")
    # Chuyển đổi ảnh thành mảng NumPy
    img_array = np.array(gray_image)
    # Áp dụng ngưỡng
    threshold = 50
    binary_array = np.where(img_array >= threshold, 255, 0).astype(np.uint8)
    # Chuyển đổi mảng NumPy trở lại thành ảnh PIL
    binary_image = Image.fromarray(binary_array, 'L')
    current_image_pil = binary_image
    display_image_in_edit_canvas(current_image_pil)
    display_image_info(current_image_pil)
def adjustment_image():
    global current_image_pil, info_frame, brightness_slider, contrast_slider, saturation_slider, adjustment_frame
    push_to_undo_stack()
    def update_slider_from_entry(slider, entry):
        #Cập nhật giá trị của slider từ giá trị nhập vào ô entry.
        value = float(entry.get())
        slider.set(value)
    def update_entry_from_slider(entry, value):
        #Cập nhật giá trị của ô entry từ giá trị thay đổi của slider.
        entry.delete(0, tk.END)
        entry.insert(0, str(value))
    def apply_adjustment():
        #Áp dụng các thay đổi về độ sáng, độ tương phản và độ bão hòa.
        brightness = int(brightness_entry.get())
        contrast = float(contrast_entry.get())
        saturation = float(saturation_entry.get())

        # Apply brightness
        enhancer = ImageEnhance.Brightness(original_image_pil)
        adjusted_image = enhancer.enhance(brightness / 100)

        # Apply contrast
        enhancer = ImageEnhance.Contrast(adjusted_image)
        adjusted_image = enhancer.enhance(contrast)

        # Apply saturation
        enhancer = ImageEnhance.Color(adjusted_image)
        adjusted_image = enhancer.enhance(saturation)

        # Cập nhật hình ảnh hiện tại
        global current_image_pil
        current_image_pil = adjusted_image
        display_image_in_edit_canvas(current_image_pil)
        display_image_info(current_image_pil)
    def reset_adjustment():
        #Đặt lại các giá trị của thanh trượt và ô nhập số về mặc định.
        brightness_slider.set(100)
        contrast_slider.set(1.0)
        saturation_slider.set(1.0)
    def close_adjustment_frame():
        global adjustment_frame
        if adjustment_frame is not None:
            adjustment_frame.destroy()  #hủy khung điều khiển resize
            adjustment_frame = None
    # Tạo khung điều chỉnh
    adjustment_frame = ttk.LabelFrame(info_frame, text="Adjustment", padding=10)
    adjustment_frame.pack(fill="x", expand=True, pady=5)
    
    # ---- Brightness ----
    Label(adjustment_frame, text="Brightness:").grid(row=0, column=0, padx=5, pady=5)
    brightness_entry = Entry(adjustment_frame, width=5)
    brightness_entry.grid(row=0, column=1, padx=5, pady=5)
    brightness_entry.insert(0, "100")
    brightness_slider = Scale(
        adjustment_frame,
        from_=0, to=200,
        orient=tk.HORIZONTAL,
        command=lambda value: update_entry_from_slider(brightness_entry, int(value))
    )
    brightness_slider.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
    brightness_slider.set(100)

    # Đồng bộ hóa ô nhập và thanh trượt
    brightness_entry.bind("<Return>", lambda event: update_slider_from_entry(brightness_slider, brightness_entry))

    # ---- Contrast ----
    Label(adjustment_frame, text="Contrast:").grid(row=2, column=0, padx=5, pady=5)
    contrast_entry = Entry(adjustment_frame, width=5)
    contrast_entry.grid(row=2, column=1, padx=5, pady=5)
    contrast_entry.insert(0, "1.0")
    contrast_slider = Scale(
        adjustment_frame,
        from_=0.1, to=2.0,
        resolution=0.1,
        orient=tk.HORIZONTAL,
        command=lambda value: update_entry_from_slider(contrast_entry, round(float(value), 1))
    )
    contrast_slider.grid(row=3, column=0, columnspan=2, padx=5, pady=5)
    contrast_slider.set(1.0)

    # Đồng bộ hóa ô nhập và thanh trượt
    contrast_entry.bind("<Return>", lambda event: update_slider_from_entry(contrast_slider, contrast_entry))

    # ---- Saturation ----
    Label(adjustment_frame, text="Saturation:").grid(row=4, column=0, padx=5, pady=5)
    saturation_entry = Entry(adjustment_frame, width=5)
    saturation_entry.grid(row=4, column=1, padx=5, pady=5)
    saturation_entry.insert(0, "1.0")
    saturation_slider = Scale(
        adjustment_frame,
        from_=0.1, to=2.0,
        resolution=0.1,
        orient=tk.HORIZONTAL,
        command=lambda value: update_entry_from_slider(saturation_entry, round(float(value), 1))
    )
    saturation_slider.grid(row=5, column=0, columnspan=2, padx=5, pady=5)
    saturation_slider.set(1.0)

    # Đồng bộ hóa ô nhập và thanh trượt
    saturation_entry.bind("<Return>", lambda event: update_slider_from_entry(saturation_slider, saturation_entry))

    # ---- Buttons ----
    
    button_frame = tk.Frame(adjustment_frame)
    button_frame.grid(row=6, column=0, columnspan=2, padx=5, pady=5)
    apply_button = Button(button_frame, text="OK", command=apply_adjustment, font=("Arial", 12, "bold"), bg="#4CAF50", fg="white")
    apply_button.pack(side=tk.LEFT, padx=5)
    reset_button = Button(button_frame, text="Reset", command=reset_adjustment, font=("Arial", 12, "bold"), bg="red", fg="white")
    reset_button.pack(side=tk.LEFT, padx=5)
    cancel_button = Button(button_frame, text="Cancel", command=close_adjustment_frame, font=("Arial", 12, "bold"), bg="gray", fg="white")
    cancel_button.pack(side=tk.LEFT, padx=5)
def edge_detection():
    global current_image_pil, original_image_pil, info_frame, edited_image

    push_to_undo_stack()  # Lưu trạng thái hiện tại để có thể quay lại nếu cần

    def apply_sobel():
        """Áp dụng bộ lọc Sobel để phát hiện biên."""
        try:
            image_gray = current_image_pil.convert("L")  # Chuyển sang grayscale
            width, height = image_gray.size
            pixels = image_gray.load()

            # Tạo ma trận cho Sobel
            gx = [[0] * width for _ in range(height)]
            gy = [[0] * width for _ in range(height)]
            img_final = [[0] * width for _ in range(height)]

            # Tính toán Sobel
            for i in range(1, height - 1):
                for j in range(1, width - 1):
                    gx[i][j] = (pixels[j-1, i-1] + 2*pixels[j-1, i] + pixels[j-1, i+1]) - \
                               (pixels[j+1, i-1] + 2*pixels[j+1, i] + pixels[j+1, i+1])
                    gy[i][j] = (pixels[j-1, i-1] + 2*pixels[j, i-1] + pixels[j+1, i-1]) - \
                               (pixels[j-1, i+1] + 2*pixels[j, i+1] + pixels[j+1, i+1])

                    magnitude = min(255, int((gx[i][j]**2 + gy[i][j]**2)**0.5))
                    img_final[i][j] = magnitude

            sobel_image = Image.new("L", (width, height))
            sobel_pixels = sobel_image.load()
            for i in range(height):
                for j in range(width):
                    sobel_pixels[j, i] = img_final[i][j]
            update_preview(sobel_image)
        except Exception as e:
            messagebox.showerror("Error", f"Could not apply Sobel edge detection: {e}")

    def apply_prewitt():
        """Áp dụng bộ lọc Prewitt để phát hiện biên."""
        try:
            image_gray = current_image_pil.convert("L")
            width, height = image_gray.size
            pixels = image_gray.load()

            # Tạo ma trận cho Prewitt
            gx = [[0] * width for _ in range(height)]
            gy = [[0] * width for _ in range(height)]
            img_final = [[0] * width for _ in range(height)]

            # Tính toán Prewitt
            for i in range(1, height - 1):
                for j in range(1, width - 1):
                    gx[i][j] = (pixels[j-1, i-1] + pixels[j-1, i] + pixels[j-1, i+1]) - \
                               (pixels[j+1, i-1] + pixels[j+1, i] + pixels[j+1, i+1])
                    gy[i][j] = (pixels[j-1, i-1] + pixels[j, i-1] + pixels[j+1, i-1]) - \
                               (pixels[j-1, i+1] + pixels[j, i+1] + pixels[j+1, i+1])

                    magnitude = min(255, int((gx[i][j]**2 + gy[i][j]**2)**0.5))
                    img_final[i][j] = magnitude

            prewitt_image = Image.new("L", (width, height))
            prewitt_pixels = prewitt_image.load()
            for i in range(height):
                for j in range(width):
                    prewitt_pixels[j, i] = img_final[i][j]
            update_preview(prewitt_image)
        except Exception as e:
            messagebox.showerror("Error", f"Could not apply Prewitt edge detection: {e}")

    def apply_canny():
        """Áp dụng thuật toán Canny để phát hiện biên mà không sử dụng thư viện."""
        try:
            # Chuyển ảnh PIL hiện tại sang grayscale dưới dạng numpy array
            img = np.array(current_image_pil.convert('L'))
            
            # Hàm Gaussian Blur
            def gaussian_blur(img, sigma=1.4):
                return gaussian_filter(img, sigma=sigma)

            # Hàm Sobel Filter
            def sobel_filter(img):
                gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
                ix = np.zeros(img.shape)
                iy = np.zeros(img.shape)

                for i in range(1, img.shape[0] - 1):
                    for j in range(1, img.shape[1] - 1):
                        ix[i, j] = np.sum(gx * img[i-1:i+2, j-1:j+2])
                        iy[i, j] = np.sum(gy * img[i-1:i+2, j-1:j+2])

                gradient_magnitude = np.sqrt(ix**2 + iy**2)
                gradient_direction = np.arctan2(iy, ix)
                return gradient_magnitude, gradient_direction

            # Hàm Non-Maximum Suppression
            def non_max_suppression(grad_mag, grad_dir):
                z = np.zeros(grad_mag.shape)
                angle = grad_dir * 180. / np.pi
                angle[angle < 0] += 180

                for i in range(1, grad_mag.shape[0] - 1):
                    for j in range(1, grad_mag.shape[1] - 1):
                        try:
                            q, r = 255, 255
                            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                                q = grad_mag[i, j + 1]
                                r = grad_mag[i, j - 1]
                            elif (22.5 <= angle[i, j] < 67.5):
                                q = grad_mag[i + 1, j - 1]
                                r = grad_mag[i - 1, j + 1]
                            elif (67.5 <= angle[i, j] < 112.5):
                                q = grad_mag[i + 1, j]
                                r = grad_mag[i - 1, j]
                            elif (112.5 <= angle[i, j] < 157.5):
                                q = grad_mag[i - 1, j - 1]
                                r = grad_mag[i + 1, j + 1]

                            if (grad_mag[i, j] >= q) and (grad_mag[i, j] >= r):
                                z[i, j] = grad_mag[i, j]
                            else:
                                z[i, j] = 0
                        except IndexError:
                            pass
                return z

            # Hàm Threshold
            def threshold(img, low, high):
                strong = 255
                weak = 25
                res = np.zeros(img.shape)
                strong_i, strong_j = np.where(img >= high)
                weak_i, weak_j = np.where((img <= high) & (img >= low))
                res[strong_i, strong_j] = strong
                res[weak_i, weak_j] = weak
                return res, weak, strong

            # Hàm Hysteresis
            def hysteresis(img, weak, strong=255):
                for i in range(1, img.shape[0] - 1):
                    for j in range(1, img.shape[1] - 1):
                        if img[i, j] == weak:
                            if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                                    or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                                    or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (img[i - 1, j + 1] == strong)):
                                img[i, j] = strong
                            else:
                                img[i, j] = 0
                return img

            # Các bước trong thuật toán Canny
            img_blur = gaussian_blur(img, sigma=1.4)
            grad_mag, grad_dir = sobel_filter(img_blur)
            non_max_img = non_max_suppression(grad_mag, grad_dir)
            thresh_img, weak, strong = threshold(non_max_img, low=20, high=50)
            img_canny = hysteresis(thresh_img, weak, strong)

            # Chuyển ảnh kết quả về PIL và cập nhật canvas
            canny_image = Image.fromarray(img_canny).convert("L")
            update_preview(canny_image)

        except Exception as e:
            messagebox.showerror("Error", f"Could not apply custom Canny edge detection: {e}")

    def apply_robert():
        """Áp dụng bộ lọc Robert để phát hiện biên."""
        try:
            image_gray = current_image_pil.convert("L")
            width, height = image_gray.size
            pixels = image_gray.load()

            # Tạo ma trận cho Robert
            gx = [[0] * width for _ in range(height)]
            gy = [[0] * width for _ in range(height)]
            img_final = [[0] * width for _ in range(height)]

            # Tính toán Robert
            for i in range(height - 1):
                for j in range(width - 1):
                    gx[i][j] = pixels[j, i] - pixels[j+1, i+1]
                    gy[i][j] = pixels[j+1, i] - pixels[j, i+1]

                    magnitude = min(255, int((gx[i][j]**2 + gy[i][j]**2)**0.5))
                    img_final[i][j] = magnitude

            robert_image = Image.new("L", (width, height))
            robert_pixels = robert_image.load()
            for i in range(height):
                for j in range(width):
                    robert_pixels[j, i] = img_final[i][j]
            update_preview(robert_image)
        except Exception as e:
            messagebox.showerror("Error", f"Could not apply Robert edge detection: {e}")

    def update_preview(new_image):
        """Cập nhật ảnh xem trước trong canvas chỉnh sửa."""
        global edited_image
        edited_image = new_image
        display_image_in_edit_canvas(edited_image)

    def apply_changes():
        """Áp dụng thay đổi lên ảnh hiện tại."""
        global current_image_pil, edited_image
        if edited_image:
            current_image_pil = edited_image
            display_image_info(current_image_pil)
        close_edge_detection()

    def reset_changes():
        """Đặt lại ảnh về trạng thái ban đầu."""
        global current_image_pil, original_image_pil
        current_image_pil = original_image_pil.copy()
        display_image_in_edit_canvas(current_image_pil)
        display_image_info(current_image_pil)

    def close_edge_detection():
        """Đóng khung chức năng Edge Detection."""
        for widget in info_frame.winfo_children():
            if isinstance(widget, ttk.LabelFrame) and widget.cget("text") == "Edge Detection":
                widget.destroy()

    # Tạo khung Edge Detection
    edge_frame = ttk.LabelFrame(info_frame, text="Edge Detection", padding=10)
    edge_frame.pack(fill="x", expand=True, pady=5)

    # Thêm các nút lựa chọn phương pháp
    ttk.Button(edge_frame, text="Sobel", command=apply_sobel).pack(fill="x", pady=5)
    ttk.Button(edge_frame, text="Prewitt", command=apply_prewitt).pack(fill="x", pady=5)
    ttk.Button(edge_frame, text="Robert", command=apply_robert).pack(fill="x", pady=5)
    ttk.Button(edge_frame, text="Canny", command=apply_canny).pack(fill="x", pady=5)

    # Thêm nút OK và Reset
    button_frame = tk.Frame(edge_frame)
    button_frame.pack(fill="x", pady=10)

    tk.Button(button_frame, text="OK", command=apply_changes, font=("Arial", 12, "bold"), bg="#4CAF50", fg="white").pack(side="left", padx=5, expand=True)
    tk.Button(button_frame, text="Reset", command=reset_changes, font=("Arial", 12, "bold"), bg="#F44336", fg="white").pack(side="left", padx=5, expand=True)
def add_noise():
    global current_image_pil, original_image_pil, noise_frame
    push_to_undo_stack()
    def apply_noise():
        global current_image_pil
        amount = float(amount_entry.get())
        noise_type = noise_var.get()

        if noise_type == "Gaussian":
            noisy_image = add_gaussian_noise(current_image_pil, amount)
        elif noise_type == "Salt & Pepper":
            noisy_image = add_salt_pepper_noise(current_image_pil, amount)
        else:
            return
        current_image_pil = noisy_image
        display_image_in_edit_canvas(current_image_pil)

    def add_gaussian_noise(image, amount):
        """Thêm nhiễu Gaussian vào ảnh."""
        img_np = np.array(image) / 255.0  # Chuyển đổi ảnh thành dạng số thực (0-1)
        noisy_img_np = random_noise(img_np, mode='gaussian', var=amount**2)  # Thêm nhiễu Gaussian
        noisy_img_np = np.clip(noisy_img_np * 255, 0, 255).astype(np.uint8)  # Chuyển về dạng 0-255
        return Image.fromarray(noisy_img_np)

    def add_salt_pepper_noise(image, amount):
        """Thêm nhiễu muối tiêu vào ảnh."""
        img_np = np.array(image) / 255.0  # Chuyển đổi ảnh thành dạng số thực (0-1)
        noisy_img_np = random_noise(img_np, mode='s&p', amount=amount)  # Thêm nhiễu Salt & Pepper
        noisy_img_np = np.clip(noisy_img_np * 255, 0, 255).astype(np.uint8)  # Chuyển về dạng 0-255
        return Image.fromarray(noisy_img_np)

    def reset_noise():
        global current_image_pil, original_image_pil
        current_image_pil = original_image_pil.copy()
        display_image_in_edit_canvas(current_image_pil)
    def close_noise_frame():
        global noise_frame
        if noise_frame is not None:
            noise_frame.destroy()  #hủy khung add noise
            noise_frame = None
    # Tạo khung "Add Noise"
    noise_frame = ttk.LabelFrame(info_frame, text="Add Noise", padding=10)
    noise_frame.pack(fill="x", expand=True, pady=5)

    # Amount entry
    tk.Label(noise_frame, text="Amount:").grid(row=0, column=0, sticky="w")
    amount_entry = tk.Entry(noise_frame, width=10)
    amount_entry.grid(row=0, column=1, padx=5)
    amount_entry.insert(0, "0.05")  # Default amount

    # Noise type dropdown
    noise_var = tk.StringVar(value="Gaussian")
    noise_options = ["Gaussian", "Salt & Pepper"]
    noise_dropdown = ttk.Combobox(noise_frame, textvariable=noise_var, values=noise_options, state="readonly")
    noise_dropdown.grid(row=1, column=0, columnspan=2, pady=(5, 10))

    # Buttons
    button_frame = tk.Frame(noise_frame)
    button_frame.grid(row=2, column=0, columnspan=2)
    ok_button = tk.Button(button_frame, text="OK", command=apply_noise, font=("Arial", 12, "bold"), bg="#4CAF50", fg="white")
    ok_button.pack(side=tk.LEFT, padx=5)
    reset_button = tk.Button(button_frame, text="Reset", command=reset_noise, font=("Arial", 12, "bold"), bg="#F44336", fg="white")
    reset_button.pack(side=tk.LEFT, padx=5)
    cancel_button = tk.Button(button_frame, text = "Cancel", command = close_noise_frame, font = ("Arial",12,"bold"))
    cancel_button.pack(side=tk.LEFT, padx = 5)
def zoom_in():
    global current_image_pil
    push_to_undo_stack()
    
    # Phóng to ảnh (tăng kích thước lên 125%)
    width, height = current_image_pil.size
    ratio = 1.5  # Tỷ lệ phóng to

    new_width = int(width * ratio)
    new_height = int(height * ratio)

    # Resize ảnh
    current_image_pil = current_image_pil.resize((new_width, new_height), Image.Resampling.BICUBIC)
    
    # Hiển thị ảnh đã zoom trên Canvas, đảm bảo ảnh nằm ở giữa
    display_image_with_scrollbars(current_image_pil)
def zoom_out():
    global current_image_pil
    push_to_undo_stack()
   
    # Thu nhỏ ảnh (giảm kích thước xuống 80%)
    width, height = current_image_pil.size
    ratio = 0.8  # Tỷ lệ thu nhỏ

    new_width = int(width * ratio)
    new_height = int(height * ratio)
    
    # Resize ảnh
    current_image_pil = current_image_pil.resize((new_width, new_height), Image.Resampling.BICUBIC)
    
    # Hiển thị ảnh đã thu nhỏ, đảm bảo ảnh nằm ở giữa
    display_image_with_scrollbars(current_image_pil)
def display_image_with_scrollbars(image):
    global edit_canvas, edit_frame
    # Xóa tất cả các phần tử trước đó trên Canvas
    edit_canvas.delete("all")
    # Lấy kích thước ảnh
    image_width, image_height = image.size
    # Lấy kích thước Canvas hiện tại
    canvas_width = edit_canvas.winfo_width()
    canvas_height = edit_canvas.winfo_height()

    # Tính toán vị trí để đặt ảnh ở giữa Canvas
    x_offset = max((canvas_width - image_width) // 2, 0)
    y_offset = max((canvas_height - image_height) // 2, 0)

    # Chuyển ảnh PIL thành PhotoImage
    photo_image = ImageTk.PhotoImage(image)
    # Thiết lập kích thước Canvas dựa trên kích thước ảnh
    edit_canvas.config(scrollregion=(0, 0, image_width, image_height))
    # Hiển thị ảnh ở vị trí giữa Canvas
    edit_canvas.create_image(x_offset, y_offset, anchor="nw", image=photo_image)
    edit_canvas.image = photo_image  # Lưu tham chiếu để tránh bị xóa bởi garbage collector

def fit_on_screen():
    global current_image_pil
    push_to_undo_stack()
    # Get the actual size of the edit_canvas
    canvas_width = edit_canvas.winfo_width()
    canvas_height = edit_canvas.winfo_height()
    # If the canvas size is not properly initialized, wait and try again
    if canvas_width == 1 and canvas_height == 1:
        root.after(100, fit_on_screen)
        return
    image_width, image_height = current_image_pil.size
    # Calculate the scaling factor to fit the image within the canvas
    scale_factor = min(canvas_width / image_width, canvas_height / image_height)
    # Calculate the new dimensions
    new_width = int(image_width * scale_factor)
    new_height = int(image_height * scale_factor)
    # Resize the image
    resized_image = current_image_pil.resize((new_width, new_height), Image.Resampling.BICUBIC)
    # Display the resized image
    display_image_in_edit_canvas(resized_image)
def filter_image():
    global current_image_pil, original_image_pil, filter_frame
    push_to_undo_stack()
    def mean_filter(data, kernel_size):
        temp = []
        kernel_halfsize = kernel_size // 2
        data_final = np.zeros((len(data), len(data[0])), dtype=np.uint8)
        for i in range(len(data)):
            for j in range(len(data[0])):
                temp = []
                for z in range(kernel_size):
                    for k in range(kernel_size):
                        neighbor_x = i + z - kernel_halfsize
                        neighbor_y = j + k - kernel_halfsize
                        if (neighbor_x < 0) or (neighbor_x >= len(data)) or (neighbor_y < 0) or (neighbor_y >= len(data[0])): 
                            temp.append(0)
                        else:
                            temp.append(data[neighbor_x][neighbor_y])
                data_final[i][j] = int(np.mean(temp))
        return data_final
    def median_filter(data, kernel_size):
        kernel_halfsize = kernel_size // 2
        data_final = np.zeros((len(data), len(data[0])), dtype=np.uint8)
        for i in range(len(data)):
            for j in range(len(data[0])):
                temp = []  # Đặt lại temp thành danh sách rỗng trước khi sử dụng
                for z in range(kernel_size):
                    for k in range(kernel_size):
                        neighbor_x = i + z - kernel_halfsize
                        neighbor_y = j + k - kernel_halfsize
                        if (neighbor_x < 0) or (neighbor_x >= len(data)) or (neighbor_y < 0) or (neighbor_y >= len(data[0])): 
                            temp.append(0)  # Giá trị ngoài biên là 0
                        else:
                            temp.append(data[neighbor_x][neighbor_y])  # Thêm giá trị pixel lân cận vào temp
                temp.sort()  # Sắp xếp danh sách
                data_final[i][j] = temp[len(temp) // 2]  # Lấy giá trị trung vị
        return data_final

    def apply_gaussian_filter():
        global current_image_pil
        push_to_undo_stack()
        img_array = np.array(current_image_pil)
        filtered_image_array = ndimage.gaussian_filter(img_array, sigma=1)
        filtered_image = Image.fromarray(filtered_image_array)
        current_image_pil = filtered_image
        display_image_in_edit_canvas(current_image_pil)
        display_image_info(current_image_pil)
    def apply_median_filter():
        global current_image_pil
        push_to_undo_stack()
        img_array = np.array(current_image_pil)
        filtered_image_array = median_filter(img_array, kernel_size=3)
        filtered_image = Image.fromarray(filtered_image_array)
        current_image_pil = filtered_image
        display_image_in_edit_canvas(current_image_pil)
        display_image_info(current_image_pil)
    def apply_mean_filter():
        global current_image_pil
        push_to_undo_stack()
        img_array = np.array(current_image_pil)
        filtered_image_array = mean_filter(img_array, kernel_size=3)
        filtered_image = Image.fromarray(filtered_image_array)
        current_image_pil = filtered_image
        display_image_in_edit_canvas(current_image_pil)
        display_image_info(current_image_pil)
    def reset_filter():
        global current_image_pil
        current_image_pil = original_image_pil.copy()
        display_image_in_edit_canvas(current_image_pil)
    def close_filter_frame():
        global filter_frame
        if filter_frame is not None:
            filter_frame.destroy()  # Hủy khung
            filter_frame = None
    filter_frame = ttk.LabelFrame(info_frame, text="Filter", padding=10)
    filter_frame.pack(fill="x", expand=True, pady=5)

    filter_var = tk.StringVar(value="Gaussian")
    filter_options = ["Gaussian", "Median", "Mean"]
    filter_dropdown = ttk.Combobox(filter_frame, textvariable=filter_var, values=filter_options, state="readonly")
    filter_dropdown.grid(row=0, column=1, columnspan=2, pady=5)

    ok_button = tk.Button(filter_frame, text="OK", font=("Arial", 12, "bold"), bg="#4CAF50", fg="white", command=lambda: apply_filter(filter_var.get()))
    ok_button.grid(row=1, column=0, padx=5, pady=5)

    reset_button = tk.Button(filter_frame, text="Reset", font=("Arial", 12, "bold"), bg="#F44336", fg="white", command=reset_filter)
    reset_button.grid(row=1, column=1, padx=5, pady=5)

    cancel_button = tk.Button(filter_frame, text="Cancel", font=("Arial", 12, "bold"), bg="#FF5722", fg="white", command=close_filter_frame)
    cancel_button.grid(row=1, column=2, padx=5, pady=5)
    def apply_filter(selected_filter):
        if selected_filter == "Gaussian":
            apply_gaussian_filter()
        elif selected_filter == "Median":
            apply_median_filter()
        elif selected_filter == "Mean":
            apply_mean_filter()
# Update Layout Function
def create_layout():
    global tool_bar, original_frame, original_label, edit_frame, edit_canvas, info_frame, info_label, size_label, mode_label, tool_buttons_frame
    # Khung tool bar
    left_frame = tk.Frame(root)
    left_frame.pack(side="left", fill="y")
    # Khung chứa ảnh gốc (fixed size)
    original_frame = tk.Frame(left_frame, bg="#DEFBFC", width=130, height=130)
    original_frame.pack(pady=10, padx=10)  # Add padding
    original_frame.pack_propagate(False)  # Ensure fixed size
    original_label = tk.Label(original_frame, bg="#eaeaea", text="No Image", anchor="center")
    original_label.pack(expand=True, fill="both")
    # Tool Bar
    tool_bar = tk.Frame(left_frame, bg="#DEFBFC", width=150, height=400)
    tool_bar.pack(fill="y", expand=True)
    tool_bar.pack_propagate(False)  # Ensure fixed size
    # Thêm các icon
    tool_buttons_frame = tk.Frame(tool_bar, bg="#DEFBFC", height=400)
    tool_buttons_frame.pack(pady=10, expand=True, fill="both")
    add_tool_buttons(tool_buttons_frame)
    # Khung chứa ảnh cần chỉnh sửa
    edit_frame = tk.Frame(root, bg="#A3D2F0")
    edit_frame.pack(side="left", expand=True, fill="both")
    # Scrollbars
    h_scrollbar = tk.Scrollbar(edit_frame, orient="horizontal")
    h_scrollbar.pack(side="bottom", fill="x")
    v_scrollbar = tk.Scrollbar(edit_frame, orient="vertical")
    v_scrollbar.pack(side="right", fill="y")
    # Canvas
    edit_canvas = tk.Canvas(edit_frame, bg="#A3D2F0", xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
    edit_canvas.pack(side="left", expand=True, fill="both")
    # Kết nối thanh cuộn với Canvas
    h_scrollbar.config(command=edit_canvas.xview)
    v_scrollbar.config(command=edit_canvas.yview)
    # Khung chứa Image Info
    info_frame = tk.Frame(root, bg="#DEFBFC", width=300)
    info_frame.pack(side="right", fill="y", padx=5, pady=5)
    info_label = tk.Label(info_frame, text="            Image Info          ", bg="#DEFBFC", fg="black", font=("Arial", 14, "bold"))
    info_label.pack(pady=10)
    # Thông tin về size và mode
    size_label = tk.Label(info_frame, text="Size:                   pixels", font=("Arial", 12), bg="#ffffff", anchor="w", relief="solid")
    size_label.pack(fill="x", padx=5, pady=5)
    mode_label = tk.Label(info_frame, text="Mode:                         ", font=("Arial", 12), bg="#ffffff", anchor="w", relief="solid")
    mode_label.pack(fill="x", padx=5, pady=5)
def push_to_undo_stack():
    """Lưu trạng thái hiện tại của ảnh vào stack undo"""
    global current_image_pil, undo_stack
    if current_image_pil:
        undo_stack.append(current_image_pil.copy())
        redo_stack.clear()  # Xóa stack redo khi có thay đổi mới

def undo():
    """Hoàn tác về trạng thái trước đó"""
    global current_image_pil, undo_stack, redo_stack
    if len(undo_stack) > 0:
        # Lưu trạng thái hiện tại vào redo stack
        redo_stack.append(current_image_pil.copy())
        # Lấy trạng thái trước từ undo stack
        current_image_pil = undo_stack.pop()
        display_image_in_edit_canvas(current_image_pil)
        display_image_info(current_image_pil)

def redo():
    """Làm lại thao tác đã hoàn tác"""
    global current_image_pil, undo_stack, redo_stack
    if len(redo_stack) > 0:
        # Lưu trạng thái hiện tại vào undo stack 
        undo_stack.append(current_image_pil.copy())
        # Lấy trạng thái tiếp theo từ redo stack
        current_image_pil = redo_stack.pop()
        display_image_in_edit_canvas(current_image_pil)
        display_image_info(current_image_pil)
def reset_image():
    """Khôi phục lại ảnh ban đầu."""
    global current_image_pil, original_image_pil
    if original_image_pil:
        push_to_undo_stack()  # Lưu trạng thái hiện tại vào undo trước khi reset
        current_image_pil = original_image_pil.copy()
        display_image_in_edit_canvas(current_image_pil)
        display_image_info(current_image_pil)
def add_tool_buttons(frame):
    undo_icon = icon("undo.png")
    redo_icon = icon("redo.png")
    reset_icon = icon("reset.png")
    crop_icon = icon("crop.png")
    resize_icon = icon("resize.png")
    zoomin_icon = icon("zoom-in.png")
    zoomout_icon = icon("zoom-out.png")
    gray_icon = icon("gray.png")
    noise_icon = icon("noise.png")
    buttons = [
        {"image": undo_icon, "command": undo},
        {"image": redo_icon, "command": redo},
        {"image": reset_icon, "command": reset_image},
        {"image": crop_icon, "command": crop_image},
        {"image": resize_icon, "command": resize_image},
        {"image": gray_icon, "command": gray_scale_image},
        {"image": noise_icon, "command": add_noise},
        {"image": zoomin_icon, "command": zoom_in},
        {"image": zoomout_icon, "command": zoom_out},
        # Có thể thêm icon ở đây nếu cần
    ]

    frame.image_refs = [undo_icon, redo_icon, reset_icon, crop_icon, resize_icon, gray_icon, noise_icon, zoomin_icon, zoomout_icon]

    for index, button in enumerate(buttons):
        row = index // 3  # Chia index cho 2 để lấy số lượng row
        col = index % 3   # Lấy phần còn lại để xác định cột (0 hoặc 1)
        tk.Button(frame, image=button["image"], command=button["command"], width=32, height=32, bg="#d4e6f1").grid(row=row, column=col, padx=5, pady=5)

# Open Image
def open_image():
    global original_image_pil, current_image_pil
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")])
    if file_path:
        image = Image.open(file_path)
        original_image_pil = image
        current_image_pil = image.copy()
        undo_stack = []  # Clear undo/redo stack
        redo_stack = []
        # Display original image
        display_original_image(image)
        # Display image on edit Canvas
        display_image_in_edit_canvas(image)
        # Update image info
        display_image_info(image)
    if hasattr(root, 'resize_width_entry'):
        reset_size()
# Display Image Info
def display_image_info(image):
    #Hiển thị thông tin kích thước và chế độ của ảnh hiện tại.
    global size_label, mode_label
    if isinstance(image, Image.Image):  # Kiểm tra xem image có phải là ảnh PIL không
        size_text = f"Size: {image.size[0]} x {image.size[1]} pixels"
        mode_text = f"Mode: {image.mode}"
        size_label.config(text=size_text)
        mode_label.config(text=mode_text)
    else:
        size_label.config(text="Size: N/A")
        mode_label.config(text="Mode: N/A")
# Display Original Image
def display_original_image(image):
    global original_label
    resized_image = image.resize((130, 130), Image.Resampling.BICUBIC)
    original_image = ImageTk.PhotoImage(resized_image)
    original_label.config(image=original_image, text="")
    original_label.image = original_image
# Display Image on Canvas
def display_image_in_edit_canvas(image):
    # Display a PIL image on the editing canvas without modifying the original PIL image state.
    global edit_canvas, displayed_image, display_width, display_height
    root.update_idletasks()  # Ensure canvas dimensions are updated
    
    # Get the canvas dimensions (use default values if dimensions are not available yet)
    canvas_width = edit_canvas.winfo_width() or 800
    canvas_height = edit_canvas.winfo_height() or 600

    # Calculate the scaling factor to fit the image within the canvas while maintaining the aspect ratio
    image_width, image_height = image.size
    scale = min(canvas_width / image_width, canvas_height / image_height)
    display_width = int(image_width * scale)
    display_height = int(image_height * scale)

    # Resize the image for display
    resized_image = image.resize((display_width, display_height), Image.Resampling.BICUBIC)
    displayed_image = ImageTk.PhotoImage(resized_image)

    # Clear existing content on the canvas and display the image at the center
    edit_canvas.delete("all")
    edit_canvas.create_image(canvas_width // 2, canvas_height // 2, anchor="center", image=displayed_image)
    edit_canvas.image = displayed_image  # Keep a reference to the image to avoid garbage collection
# Save Image
def save_image():
    global current_image_pil
    file_path = filedialog.asksaveasfilename(defaultextension=".png",filetypes=[("PNG files", "*.png"),("JPEG files", "*.jpg"),("All files", "*.*")])
    if file_path:
        current_image_pil.save(file_path)
        messagebox.showinfo("Success", f"Image saved to {file_path}")
# Create Menu Bar and Layout
create_menu_bar()
create_layout()
root.mainloop()
