# importing libraries
import shutil
from tkinter import messagebox

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from tools.easy_qt import *

# importing models
model_bone = YOLO("models/hipbone_segmentation.pt")
model_stress = YOLO(r"models/fracture_detector.pt")
model_femur = YOLO(r"models/hipbone_segmentation.pt")

def zoom_highest_prob_fracture(image_path, detections, zoom_factor=4):
    """ purpose: to show the fracture with the highest probability to highlight"""
    if not detections:
        print("No detections.")
        return None
    # best = highest-confidence
    best = max(detections, key=lambda d: d[4])
    x1, y1, x2, y2, conf, class_id = best

    # get image
    original = Image.open(image_path)
    width, height = original.size
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(width, int(x2)), min(height, int(y2))

    # crop/zoom
    cropped = original.crop((x1, y1, x2, y2))
    zoomed = cropped.resize((cropped.width * zoom_factor, cropped.height * zoom_factor))

    return ImageTk.PhotoImage(zoomed)

class BoneOptionMenu(tk.OptionMenu):
    """ drop-down menu to choose bone type (bold font for existing ones)"""
    def __init__(self, master, variable, values, parts_found=None, **kwargs):
        super().__init__(master, variable, *values, **kwargs)
        self.variable = variable
        self.parts_found = parts_found or []
        self['font'] = ("Tahoma", 12)
        self.menu = self['menu']
        self._update_menu(values)
        self.fracture_image_label = None
        self.variable.trace_add("write", self._update_selected_bold)
        self.config(bg="#f0f0f0", activebackground="#c4c4c4", relief="groove", bd=2)

    def _update_menu(self, values):
        self.menu.delete(0, "end")
        for val in values:
            def _cmd(v=val):
                self.variable.set(v)
            font_style = ("Tahoma", 12, "bold") if val in self.parts_found else ("Tahoma", 12)
            self.menu.add_command(label=val, command=_cmd, font=font_style)

    def _update_selected_bold(self):
        for index in range(self.menu.index("end")+1):
            label = self.menu.entrycget(index, "label")
            font_style = ("Tahoma", 12, "bold") if label in self.parts_found else ("Tahoma", 12)
            self.menu.entryconfig(index, font=font_style)

def get_output_structure(image_path):
    """directory structure def"""
    base_dir = os.path.join(os.path.dirname(image_path), "segmented")
    return {
        "base": base_dir,
        "parts": os.path.join(base_dir, "parts"),
        "masks": os.path.join(base_dir, "masks"),
        "full_results": os.path.join(base_dir, "full_results"),
        "fractures": os.path.join(base_dir, "fractures")  # Added missing key
    }


def create_directories(output_dirs):
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)


def clear_directory(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def analyze_parts(image_path):
    output_dirs = get_output_structure(image_path)
    for dir_type in ["parts", "masks", "fractures"]:
        clear_directory(output_dirs[dir_type])
    # refresh
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    try:
        # load/run prediction
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Resim Bulunamadı: {image_path}")
        results = model_bone.predict(image_path, conf=0.15)[0]
        # full segmentation result
        full_segmented_path = os.path.join(output_dirs["full_results"], "full_segmented.jpg")
        results.save(filename=full_segmented_path)

        # process masks (available masks)
        if results.masks is None:
            print("[!] No masks detected in the image")
            return False
        masks = results.masks.data
        boxes = results.boxes.xyxy
        names = results.names
        classes = results.boxes.cls

        for i, (mask, box, cls_idx) in enumerate(zip(masks, boxes, classes)):
            label = names[int(cls_idx)]
            mask_np = (mask.cpu().numpy() > 0.5).astype(np.uint8)
            # mask matches image dimensions - check
            if mask_np.shape != image.shape[:2]:
                mask_np = cv2.resize(mask_np, (image.shape[1], image.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

            masked_image = cv2.bitwise_and(image, image, mask=mask_np) # mask to original

            # ROI coordinates
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

            # crop/save
            cropped_image = masked_image[y1:y2, x1:x2]
            cropped_mask = mask_np[y1:y2, x1:x2] * 255

            if cropped_image.size == 0:
                print(f"[!] Empty crop for {label}_{i+1}")
                continue
            base_name = f"{label}_{i+1}"
            cv2.imwrite(os.path.join(output_dirs["parts"], f"{base_name}.png"), cropped_image)
            cv2.imwrite(os.path.join(output_dirs["masks"], f"{base_name}_mask.png"), cropped_mask)
        return True

    except Exception as e:
        print(f"[!] Critical error in analysis: {str(e)}")
        return False


def adjust_contrast(image, alpha=2.0, beta=0):
    img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    adjusted = cv2.convertScaleAbs(img_np, alpha=alpha, beta=beta)
    return Image.fromarray(cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB))


def hairline_check(image_path, conf_threshold=0.15):
    try:
        results = model_stress.predict(image_path, conf=conf_threshold)
        if results and results[0].boxes:
            return results[0]

        # retry - with contrast
        img = Image.open(image_path).convert("RGB")
        img_contrast = adjust_contrast(img, alpha=2.0)
        tmp_path = os.path.join(os.path.dirname(image_path), "contrast_tmp.jpg")
        img_contrast.save(tmp_path)
        results_retry = model_stress.predict(tmp_path, conf=0.15)
        os.remove(tmp_path)
        return results_retry[0] if results_retry and results_retry[0].boxes else None

    except Exception as e:
        print(f"Stres kırığı tespitinde hata ile karşılaşıldı.: {e}")
        return None


def save_fracture_images(segmented_dir, output_dir):
    detected_files = []
    if not os.path.exists(segmented_dir):
        print(f"klasör bulunamadı: {segmented_dir}")
        return detected_files
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    try:
        for file in os.listdir(segmented_dir):
            if file.endswith(".png") and "_mask" not in file:
                full_path = os.path.join(segmented_dir, file)
                detection = hairline_check(full_path)
                if detection and detection.masks is not None:
                    img = Image.open(full_path).convert("RGB")
                    img_np = np.array(img)
                    img_h, img_w = img_np.shape[:2]
                    masks = detection.masks.data.cpu().numpy()
                    for mask in masks:
                        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                        colored_mask = np.zeros_like(img_np)
                        colored_mask[mask > 0.5] = [255, 0, 0]
                        img_np = cv2.addWeighted(img_np, 1, colored_mask, 0.4, 0)
                    result_img = Image.fromarray(img_np)
                    save_path = os.path.join(output_dir, file)
                    result_img.save(save_path)
                    detected_files.append((file, save_path))
    except Exception as e:
        print(f"Kırıklı görüntüler işlenirken hata ile karşılaşıldı.: {e}")
    return detected_files


def calculate_size_classification(length):
    print("Uzunluk:", length)
    if length > 450:
        return "Uzun", "#ff0000"
    elif 300 <= length <= 650:
        return "Orta", "#ffa500"
    else:
        return "Kısa", "#00ff00"

def calculate_contrast_sensitivity(enhanced_gray, thresh):
    fracture_region = enhanced_gray[thresh == 255]
    surrounding_bone = enhanced_gray[(thresh != 255) & (enhanced_gray> 0)]
    return np.mean(fracture_region) - np.mean(surrounding_bone)


class HairlineUI:
    OPTIONS = [
        "AcetabulumL","AcetabulumR","FemurL","FemurR","GreaterTrochanterL","GreaterTrochanterR",
        "IliumL","IliumR","IschiumL","IschiumR","LesserTrochanterL","LesserTrochanterR",
        "PelvicBrim","PubisL","PubisR","Sacrum&Coccyx","TearDropL","TearDropR"
    ]
    def __init__(self, root, assets, callbacks):
        self.root = root
        self.root.attributes('-fullscreen', True)
        self.assets = assets
        self.callbacks = callbacks
        self.selected_option = tk.StringVar(value=self.OPTIONS[0])
        create_image(root, self.assets['background_path'], 0, 0)
        self.image_canvas = tk.Label(root)
        self.image_canvas.place(x=48, y=72, width=853, height=719)
        create_button_with_image(
            root,
            self.assets['upload'],
            700, 904,
            self.upload_new_image
        )
        self.original_image_path = assets['example_image']
        # fracture analysis panel (right section)
        self.analysis_frame = tk.Frame(root, bg="white", width=357, height=491)
        self.analysis_frame.place(x=1517, y=91)
        self.fracture_image_label = None
        self.show_overlay = True
        self.toggle_btn = create_fancy_text_button(
            root, "ORİJİNALİ GÖSTER", 1600, 500, self.toggle_overlay
        )
        self.create_analysis_widgets()
        output_dirs = get_output_structure(self.original_image_path)
        parts_dir = output_dirs["parts"]
        parts_found = []
        if os.path.exists(parts_dir):
            for part in self.OPTIONS:
                if any(fname.startswith(part) for fname in os.listdir(parts_dir)):
                    parts_found.append(part)
        self.dropdown = BoneOptionMenu(root, self.selected_option, self.OPTIONS, parts_found)
        self.dropdown.place(x=39, y=806)
        create_fancy_text_button(root, "GÖSTER", 200, 806, self.on_show_part)
        create_button_with_image(root, self.assets['risk_area_analysis'], 650, 806, self.show_fracture_analysis)
        create_button_with_image(root, self.assets['download'], 335, 905, self.callbacks.get('download_results', lambda: None))
        create_button_with_image(root, self.assets['refreash'], 525, 904, self.callbacks.get('clear_results', lambda: None))
        create_button_with_image(root, self.assets['close'], 1767, 943, self.callbacks.get('close_app', lambda: None))
        create_button_with_image(root, self.assets['load'], 510, 804, self.load_and_analyze)
        create_fancy_text_button(root, "ORİJİNAL", 320, 806, self.display_original)
        self.display_original()
        self.fracture_frame = tk.Frame(root, width=540, height=380)
        self.fracture_frame.place(x=950, y=190)
        self.current_fracture_index = 0
        self.detected_fractures = []
        self.fracture_heading = tk.Label(root, font=("Tahoma", 14, "bold"), bg="white")
        self.fracture_heading.place(x=1030, y=126)
        create_button_with_image(root, assets.get('back'), 956, 100, self.prev_fracture)
        create_button_with_image(root, assets.get('forward'), 1399, 100, self.next_fracture)
        # labels
        self.total_fracture_label = tk.Label(root, text="0", font=("Tahoma", 12, "bold"), bg="white")
        self.total_fracture_label.place(x=127, y=900)
        self.highest_severity_label = tk.Label(root, text="Yok", font=("Tahoma", 12, "bold"), bg="white")
        self.highest_severity_label.place(x=149, y=979)
        self.zoomed_frame = tk.Frame(root, bg="white", width=540, height=380)
        self.zoomed_frame.place(x=963, y=681)
        self.zoomed_label = tk.Label(self.zoomed_frame)
        self.zoomed_label.pack()

    def show_uploaded_image(self, path):
        try:
            img = Image.open(path)
            img = img.resize((853, 719))
            self.photo = ImageTk.PhotoImage(img)
            self.image_canvas.config(image=self.photo)
            self.image_canvas.image = self.photo
        except Exception as e:
            print(f"Resim gösterilemiyor: {e}")


    def create_analysis_widgets(self):
        self.metrics = {
            'boyut': tk.StringVar(value="Boyut: Hesaplanıyor..."),
            'şiddet': tk.StringVar(value="Şiddet: Belirsiz"),
            'olasılık': tk.StringVar(value="Olasılık: %--"),
            'konum': tk.StringVar(value="Konum: Belirsiz")
        }
        self.recommendations = tk.Text(
            self.analysis_frame, wrap=tk.WORD,
            width=40, height=10, font=("Tahoma", 11), bg="#f0f0f0")
        self.recommendations.pack(pady=10)
        metrics_frame = tk.Frame(self.analysis_frame, bg="white")
        self.severity_labels = {}
        severity_colors = {
            'boyut': "#FFFFFF",
            'şiddet': "#FFFFFF",
            'olasılık': "#FFFFFF",
            'konum': "#FFFFFF"
        }
        for i, (key, var) in enumerate(self.metrics.items()):
            lbl = tk.Label(metrics_frame, textvariable=var,
                         font=("Tahoma", 12, "bold"), bg=severity_colors[key],
                         padx=10, pady=2)
            lbl.grid(row=i, column=0, sticky="ew", pady=5)
            self.severity_labels[key] = lbl
        metrics_frame.pack(fill='x')


    def toggle_overlay(self):
        self.show_overlay = not self.show_overlay
        self.toggle_btn.config(text="ORİJİNAL GÖSTER" if self.show_overlay else "KIRIK GÖSTER")
        self.update_fracture_display()


    def display_fracture_image(self, img):
        if self.fracture_image_label:
            self.fracture_image_label.destroy()
        img = img.resize((500, 400), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.fracture_image_label = tk.Label(self.fracture_frame, image=photo)
        self.fracture_image_label.image = photo  # Keep reference
        self.fracture_image_label.pack()


    def update_fracture_display(self):
        if self.detected_fractures:
            fracture_data = self.detected_fractures[self.current_fracture_index]
            if self.show_overlay:
                img = Image.open(fracture_data['overlay_path'])
            else:
                img = Image.open(fracture_data['original_path'])
            self.display_fracture_image(img)
            self.update_metrics_from_cache(fracture_data)


    def update_metrics_from_cache(self, fracture_data):
        self.metrics['boyut'].set(f"Boyut: {fracture_data['size_class']}")
        self.severity_labels['boyut'].config(bg=fracture_data['size_color'])

        self.metrics['şiddet'].set(f"Şiddet: {fracture_data['severity_class']}")
        self.severity_labels['şiddet'].config(bg=fracture_data['severity_color'])

        current_prob = self.get_fracture_probability(fracture_data['file_name'])
        self.metrics['olasılık'].set(f"Olasılık: %{current_prob:.1f}")

        self.metrics['konum'].set(f"Konum: {fracture_data['location']}")

        self.update_recommendations(fracture_data['length'],
                           fracture_data['contrast_diff'],
                           fracture_data['location'])

    def calculate_medical_metrics(self, image, fracture_data):  # Add fracture_data parameter
        img_np = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced_gray = clahe.apply(gray)
        _, thresh = cv2.threshold(enhanced_gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_length = 0
        for cnt in contours:
            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                major_axis = max(ellipse[1])
                max_length = max(max_length, major_axis)
        prob = self.get_fracture_probability(fracture_data['file_name'])
        size_text, color = calculate_size_classification(max_length)
        contrast_diff = calculate_contrast_sensitivity(enhanced_gray, thresh)
        severity, severity_color = self.calculate_severity_classification(contrast_diff)
        # update
        fracture_data.update({
            'length': max_length,
            'contrast_diff': contrast_diff,
            'probability': prob,
            'size_class': size_text,
            'size_color': color,
            'severity_class': severity,
            'severity_color': severity_color,
            'location': fracture_data['file_name'].split('_')[0]
        })
        self.update_recommendations(max_length, contrast_diff, fracture_data['location'])


    def calculate_severity_classification(self, contrast_diff):
        print("Kontrast Farkı", contrast_diff)
        if contrast_diff < -90 or contrast_diff > 90:
            return "Kritik", "#ff0000"
        elif -50 <= contrast_diff < -70 or contrast_diff > 70:
            return "Orta", "#ffa500"
        else:
            return "Hafif", "#00ff00"

    def update_size_metric(self, length):
        if length > 200:
            size_text = "Uzun"
            color = "#ff0000"
        elif 100 <= length <= 200:
            size_text = "Orta"
            color = "#ffa500"
        else:
            size_text = "Kısa"
            color = "#00ff00"

        self.metrics['boyut'].set(f"Boyut: {size_text}")
        self.severity_labels['boyut'].config(bg=color)

    def update_severity_metric(self, contrast_diff):
        if contrast_diff < -90 or contrast_diff > 90:
            severity = "Kritik"
            color = "#ff0000"
        elif -70 <= contrast_diff < -50 or contrast_diff > 70:
            severity = "Orta"
            color = "#ffa500"
        else:
            severity = "Hafif"
            color = "#00ff00"

        self.metrics['şiddet'].set(f"Şiddet: {severity}")
        self.severity_labels['şiddet'].config(bg=color)


    def get_fracture_probability(self, file_name):
        part_path = os.path.join(
            get_output_structure(self.original_image_path)["parts"],
            file_name
        )

        if not os.path.exists(part_path):
            print(f"Part image missing: {part_path}")
            return 0.0

        try:
            results = model_stress.predict(part_path, conf=0.01)  # threshold
            if not results or not results[0].boxes:
                return 0.0
            confidences = results[0].boxes.conf.cpu().numpy()
            max_conf = np.max(confidences) if len(confidences) > 0 else 0
            return max_conf * 100
        except Exception as e:
            print(f"Probability error in {file_name}: {str(e)}")
            return 0.0

    def update_recommendations(self, length, contrast, location):
        text = "Klinik Öneriler:\n\n"
        text += f"{location} bölgesi için öneriler:\n"
        # length recommendations
        if length > 400:
            text += "- Acil cerrahi müdahale gereklidir\n"
            text += "- Yüksek riskli instabil kırık\n"
        elif 120 <= length <= 400:
            text += "- Hasta immobilizasyon ile takip edilmeli\n"
            text += "- Haftalık radyografik kontrol\n"
        else:
            text += "- Analjezik tedavi ve istirahat öner\n"
            text += "- 3 gün sonra kontrole çağır\n"
        # contrast-based recommendations
        if contrast < -120 or contrast > 120:
            text += "\nAkut travma bulguları: Travma öyküsü araştır\n"
            text += "NSAID tedavisi başlanmalı"
        elif -40 <= contrast < -25 or contrast > 40:
            text += "\nSubakut bulgular: Fizik tedavi öner\n"
        else:
            text += "\nKronik değişiklikler: Kemik yoğunluğu ölçümü yap"
        self.recommendations.delete(1.0, tk.END)
        self.recommendations.insert(tk.END, text)

    def upload_new_image(self):
        file_path = filedialog.askopenfilename(
            title="Choose an Image",
            filetypes=[("Image files", "*.jpg *.png *.jpeg")]
        )
        if file_path:
            self.original_image_path = file_path
            success = analyze_parts(self.original_image_path)
            if success:
                self.refresh_dropdown_bolding()
                self.clear_previous_results()
                self.display_original()
                messagebox.showinfo("Başarılı", "Yeni resim yüklendi ve analiz edildi!")
            else:
                messagebox.showwarning("Uyarı", "Analiz gerçekleştirilemedi.")

    def refresh_dropdown_bolding(self):
        output_dirs = get_output_structure(self.original_image_path)
        parts_dir = output_dirs["parts"]
        updated_parts_found = []
        if os.path.exists(parts_dir):
            for part in self.OPTIONS:
                if any(fname.startswith(part) for fname in os.listdir(parts_dir)):
                    updated_parts_found.append(part)
        self.dropdown.parts_found = updated_parts_found
        self.dropdown._update_menu(self.OPTIONS)

    def show_fracture_analysis(self):
        output_dirs = get_output_structure(self.original_image_path)
        parts_dir = output_dirs["parts"]
        fractures_dir = output_dirs["fractures"]

        if not os.path.exists(parts_dir):
            messagebox.showerror("Hata", "Önce görüntüyü analiz edin!")
            return

        raw_detections = save_fracture_images(parts_dir, fractures_dir)
        self.detected_fractures = []

        for file_name, image_path in raw_detections:
            bone_name = file_name.split('_')[0]
            part_path = os.path.join(parts_dir, file_name)

            fracture_data = {
                'file_name': file_name,
                'overlay_path': image_path,
                'original_path': os.path.join(parts_dir, file_name),
                'location': bone_name,
                'probability': self.get_fracture_probability(file_name),
                'length': 0,
                'contrast_diff': 0,
                'size_class': "Belirsiz",
                'size_color': "#FFFFFF",
                'severity_class': "Belirsiz",
                'severity_color': "#FFFFFF"
            }

            img = Image.open(part_path)
            self.calculate_medical_metrics(img, fracture_data)
            self.detected_fractures.append(fracture_data)

        if not self.detected_fractures:
            messagebox.showinfo("Bilgi", "Hiç stres kırığı tespit edilmedi")
            return

        self.total_fracture_label.config(text=f"{len(self.detected_fractures)}")

        severity_order = {"Kritik": 3, "Orta": 2, "Hafif": 1}
        if self.detected_fractures:
            highest = max(self.detected_fractures, key=lambda x: severity_order.get(x['severity_class'], 0))
            self.highest_severity_label.config(
                text=f"{highest['severity_class']}",
                bg=highest['severity_color']
            )
        else:
            self.highest_severity_label.config(text="En Yüksek Risk: Yok", bg="white")

        self.current_fracture_index = 0
        most_probable = max(self.detected_fractures, key=lambda x: x['probability'])
        original_img = Image.open(most_probable['original_path']).convert('RGB')
        img_np = np.array(original_img)
        h, w = img_np.shape[:2]

        detection = hairline_check(most_probable['overlay_path'])
        if detection and detection.masks is not None and detection.masks.shape[0] > 0:
            mask = detection.masks.data[0].cpu().numpy()
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            ys, xs = np.where(mask_resized > 0.5)

            if len(xs) > 0 and len(ys) > 0:
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                zoom_factor = 4
                zoom_width, zoom_height = w // zoom_factor, h // zoom_factor

                x1 = max(0, min(w - zoom_width, cx - zoom_width // 2))
                y1 = max(0, min(h - zoom_height, cy - zoom_height // 2))
                x2 = x1 + zoom_width
                y2 = y1 + zoom_height

                cropped = img_np[y1:y2, x1:x2]

                cropped_mask = mask_resized[y1:y2, x1:x2]
                colored_mask = np.zeros_like(cropped)
                colored_mask[cropped_mask > 0.5] = [255, 0, 0]

                blended = cv2.addWeighted(cropped, 1.0, colored_mask, 0.4, 0)

                print(f"Image size: {w}x{h}")
                print(f"Zoom window size: {zoom_width}x{zoom_height}")
                print(f"Cropping area: ({x1},{y1}) to ({x2},{y2})")
                print(f"Fracture center: ({cx},{cy})")


                resized_zoom = cv2.resize(blended, (650, 320), interpolation=cv2.INTER_CUBIC)
                img_overlay = Image.fromarray(resized_zoom)
            else:
                img_overlay = original_img.resize((650, 320), Image.Resampling.LANCZOS)
        else:
            img_overlay = original_img.resize((650, 320), Image.Resampling.LANCZOS)


        photo = ImageTk.PhotoImage(img_overlay)
        self.zoomed_label.config(image=photo)
        self.zoomed_label.image = photo

        self.show_current_fracture()

    def show_current_fracture(self):
        if not self.detected_fractures:
            return
        fracture_data = self.detected_fractures[self.current_fracture_index]
        self.fracture_heading.config(text=f"Bölge: {fracture_data['location']}")
        self.update_metrics_from_cache(fracture_data)
        img_path = fracture_data['overlay_path'] if self.show_overlay else fracture_data['original_path']
        img = Image.open(img_path).resize((500,400), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)

        if self.fracture_image_label:
            self.fracture_image_label.destroy()
        self.fracture_image_label = tk.Label(self.fracture_frame, image=photo)
        self.fracture_image_label.image = photo
        self.fracture_image_label.pack()
    def next_fracture(self):
        if self.detected_fractures:
            self.current_fracture_index = (self.current_fracture_index + 1) % len(self.detected_fractures)
            self.show_current_fracture()
    def clear_previous_results(self):
        self.detected_fractures = []
        self.current_fracture_index = 0
        for widget in self.fracture_frame.winfo_children():
            widget.destroy()
        self.fracture_heading.config(text="")
    def prev_fracture(self):
        if self.detected_fractures:
            self.current_fracture_index = (self.current_fracture_index - 1) % len(self.detected_fractures)
            self.show_current_fracture()
    def set_image(self, path):
        try:
            img = Image.open(path).resize((800, 800), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.image_canvas.config(image=photo)
            self.image_canvas.image = photo
        except Exception as e:
            messagebox.showerror("Hata", f"Resim yükleme hatası: {path}\n{e}")
    def on_show_part(self):
        part = self.selected_option.get()
        output_dirs = get_output_structure(self.original_image_path)
        parts_dir = output_dirs["parts"]
        if not os.path.exists(parts_dir):
            messagebox.showerror("Hata", "Parça klasörü bulunamadı.")
            return
        matching_file = next(
            (f for f in os.listdir(parts_dir)
             if f.startswith(part + "_") and f.endswith(".png") and "_mask" not in f),
            None
        )
        if not matching_file:
            messagebox.showwarning("Uyarı", f"Bu part tespit edilemedi: {part}")
            return
        try:
            self.set_image(os.path.join(parts_dir, matching_file))
        except Exception as e:
            messagebox.showerror("Hata", f"Bu part resmi yüklenemedi: {e}")

    def display_original(self):
        if self.original_image_path:
            self.set_image(self.original_image_path)

    def load_and_analyze(self):
        if analyze_parts(self.original_image_path):
            self.refresh_dropdown_bolding()
            messagebox.showinfo("Başarılı", "Resim yüklendi ve parçalar tespit edildi!")
        else:
            messagebox.showwarning("Uyarı", "Part tespit edilemedi ya da resim yüklenmedi.")


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1000x1000")
    root.title("Bone Part Analyzer")
    assets = {
        "background_path": r"assets/hairline3.png",
        "risk_area_analysis": r"assets/riskareaanalysis.png",
        "download": r"assets/download.png",
        "refreash": r"assets/refreash.png",
        "close": "assets/close.png",
        "load": r"assets/SHOWALL.png",
        "upload": r"assets/upload.png",
        "back": r"assets/back.png",
        "forward": r"assets/forward.png",
        "example_image": r"images/examples/example.jpg"
    }
    callbacks = {
        "download_results": lambda: None,
        "clear_results": lambda: None,
        "close_app": root.quit
    }
    app = HairlineUI(root, assets, callbacks)
    root.mainloop()