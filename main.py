"""
main.py
=======
Image Processing Desktop Application
Built with Python + Tkinter + OpenCV

Run:  python main.py
"""

import tkinter as tk
from tkinter import ttk, font
import cv2
import numpy as np
import threading

import image_processing as ip
import utils


# ─────────────────────────────────────────────────────────────
#  OPERATION REGISTRY
#  Each entry: "Display Name" → (function, {param_name: default_value})
#  param_name must match the function argument (after 'image')
# ─────────────────────────────────────────────────────────────

OPERATIONS = {
    # ── Linear Filters ──────────────────────────────────────
    "Linear Filters": {
        "Mean / Box Filter":            (ip.mean_box_filter,            {"ksize": 5}),
        "Gaussian Filter":              (ip.gaussian_filter,             {"ksize": 5, "sigma": 1.0}),
        "Midpoint Filter":              (ip.midpoint_filter,             {"ksize": 5}),
        "Alpha-Trimmed Mean Filter":    (ip.alpha_trimmed_mean_filter,   {"ksize": 5, "d": 2}),
        "Harmonic Mean Filter":         (ip.harmonic_mean_filter,        {"ksize": 3}),
        "Contraharmonic Mean Filter":   (ip.contraharmonic_mean_filter,  {"ksize": 3, "Q": 1.5}),
        "Low-Pass Filter":              (ip.low_pass_filter,             {"cutoff": 30}),
        "High-Pass Filter":             (ip.high_pass_filter,            {"cutoff": 30}),
    },
    # ── Non-Linear Filters ──────────────────────────────────
    "Non-Linear Filters": {
        "Median Filter":                (ip.median_filter,               {"ksize": 5}),
        "Mode Filter":                  (ip.mode_filter,                 {"ksize": 5}),
        "Maximum Filter (Dilation)":    (ip.maximum_filter_dilation,     {"ksize": 5}),
        "Minimum Filter (Erosion)":     (ip.minimum_filter_erosion,      {"ksize": 5}),
    },
    # ── Edge Detection ──────────────────────────────────────
    "Edge Detection": {
        "Laplacian Filter":             (ip.laplacian_filter,            {}),
        "Sobel Filter":                 (ip.sobel_filter,                {}),
        "Canny Edge Detector":          (ip.canny_edge_detector,         {"low": 50, "high": 150}),
        "Prewitt Filter":               (ip.prewitt_filter,              {}),
        "Roberts Edge":                 (ip.roberts_edge,                {}),
    },
    # ── Segmentation ────────────────────────────────────────
    "Segmentation": {
        "Global Threshold":             (ip.global_threshold,            {"thresh": 127}),
        "Otsu Threshold":               (ip.otsu_threshold,              {}),
        "Adaptive Threshold":           (ip.adaptive_threshold,          {"block_size": 11, "C": 2}),
        "Multi-Otsu Threshold":         (ip.multi_otsu_threshold,        {"classes": 3}),
        "Region Growing":               (ip.region_growing,              {}),
        "Split and Merge":              (ip.split_and_merge,             {}),
        "K-Means Segmentation":         (ip.kmeans_segmentation,         {"k": 3}),
        "Watershed":                    (ip.watershed_segmentation,      {}),
        "SLIC Superpixels":             (ip.slic_superpixels,            {"n_segments": 100}),
        "Felzenszwalb":                 (ip.felzenszwalb_segmentation,   {"scale": 100}),
    },
    # ── Compression ─────────────────────────────────────────
    "Compression": {
        "RLE Compression":              (ip.rle_compress,                {}),
        "Huffman Coding":               (ip.huffman_compress,            {}),
        "LZW Compression":              (ip.lzw_compress,                {}),
        "DPCM Compression":             (ip.dpcm_compress,               {}),
        "Arithmetic Coding":            (ip.arithmetic_compress,         {}),
        "JPEG Compression":             (ip.jpeg_compression,            {"quality": 30}),
    },
    # ── Padding ─────────────────────────────────────────────
    "Padding": {
        "Zero Padding":                 (ip.zero_padding,                {"pad": 30}),
        "Replicate / Edge Padding":     (ip.replicate_padding,           {"pad": 30}),
        "Reflect Padding":              (ip.reflect_padding,             {"pad": 30}),
        "Symmetric Padding":            (ip.symmetric_padding,           {"pad": 30}),
        "Wrap / Periodic Padding":      (ip.wrap_padding,                {"pad": 30}),
        "Asymmetric Padding":           (ip.asymmetric_padding,          {"top": 10, "bottom": 40, "left": 20, "right": 60}),
    },
    # ── Extras ──────────────────────────────────────────────
    "Extra Operations": {
        "Grayscale":                    (ip.to_grayscale,                {}),
        "Histogram Equalization":       (ip.histogram_equalization,      {}),
    },
}


# ─────────────────────────────────────────────────────────────
#  COLOUR PALETTE  (dark professional theme)
# ─────────────────────────────────────────────────────────────

C = {
    "bg":        "#0f1117",
    "panel":     "#1a1d27",
    "sidebar":   "#13151f",
    "card":      "#1e2132",
    "accent":    "#4f8ef7",
    "accent2":   "#7c5cbf",
    "success":   "#3ecf8e",
    "warning":   "#f59e0b",
    "danger":    "#ef4444",
    "text":      "#e8eaf0",
    "subtext":   "#7b8099",
    "border":    "#2a2d3e",
    "hover":     "#252840",
    "slider_bg": "#2a2d3e",
}


# ─────────────────────────────────────────────────────────────
#  MAIN APPLICATION CLASS
# ─────────────────────────────────────────────────────────────

class ImageProcessingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Processing Lab")
        self.geometry("1380x820")
        self.minsize(1100, 700)
        self.configure(bg=C["bg"])

        # State
        self.original_image = None
        self.result_image   = None
        self.current_op     = tk.StringVar(value="")
        self.current_group  = tk.StringVar(value="Linear Filters")
        self._param_vars    = {}          # param_name → tk.Variable
        self._param_widgets = {}          # param_name → widget
        self._tk_orig       = None        # Keep PhotoImage reference
        self._tk_res        = None

        self._build_ui()
        self._apply_ttk_styles()

    # ── UI CONSTRUCTION ──────────────────────────────────────

    def _build_ui(self):
        """Assemble all major UI sections."""
        self._build_topbar()
        main = tk.Frame(self, bg=C["bg"])
        main.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self._build_sidebar(main)
        self._build_workspace(main)

    def _build_topbar(self):
        bar = tk.Frame(self, bg=C["panel"], height=56)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        # Logo / title
        tk.Label(bar, text="⬡", font=("Courier", 22, "bold"),
                 bg=C["panel"], fg=C["accent"]).pack(side="left", padx=(18, 6), pady=8)
        tk.Label(bar, text="Image Processing Lab",
                 font=("Segoe UI", 14, "bold"),
                 bg=C["panel"], fg=C["text"]).pack(side="left")
        tk.Label(bar, text="Python · OpenCV · NumPy",
                 font=("Segoe UI", 9), bg=C["panel"],
                 fg=C["subtext"]).pack(side="left", padx=14)

        # Top-right buttons
        btn_frame = tk.Frame(bar, bg=C["panel"])
        btn_frame.pack(side="right", padx=12)
        self._btn(btn_frame, "  📂  Open Image", self.load_image,
                  C["accent"], side="left", padx=4)
        self._btn(btn_frame, "  💾  Save Result", self.save_result,
                  C["success"], side="left", padx=4)
        self._btn(btn_frame, "  🔄  Reset", self.reset,
                  C["border"], side="left", padx=4)

        # Status bar
        self.status_var = tk.StringVar(value="No image loaded — open an image to begin.")
        tk.Label(bar, textvariable=self.status_var, font=("Segoe UI", 9),
                 bg=C["panel"], fg=C["subtext"]).pack(side="right", padx=20)

    def _build_sidebar(self, parent):
        """Left control panel: group selector + operation list + parameter sliders."""
        sidebar = tk.Frame(parent, bg=C["sidebar"], width=310)
        sidebar.pack(side="left", fill="y", padx=(0, 8), pady=0)
        sidebar.pack_propagate(False)

        # ── Section label
        tk.Label(sidebar, text="OPERATIONS", font=("Segoe UI", 8, "bold"),
                 bg=C["sidebar"], fg=C["subtext"]).pack(anchor="w", padx=16, pady=(14, 4))

        # ── Category tabs (scrollable row)
        tab_outer = tk.Frame(sidebar, bg=C["sidebar"])
        tab_outer.pack(fill="x", padx=8, pady=(0, 8))
        self._tab_buttons = {}
        groups = list(OPERATIONS.keys())
        for g in groups:
            btn = tk.Label(tab_outer, text=g, font=("Segoe UI", 8, "bold"),
                           bg=C["border"], fg=C["subtext"],
                           padx=8, pady=4, cursor="hand2")
            btn.pack(side="left", padx=2, pady=2)
            btn.bind("<Button-1>", lambda e, grp=g: self._select_group(grp))
            self._tab_buttons[g] = btn

        # ── Operation listbox
        list_frame = tk.Frame(sidebar, bg=C["sidebar"])
        list_frame.pack(fill="both", expand=True, padx=8)

        scrollbar = tk.Scrollbar(list_frame, bg=C["border"], troughcolor=C["bg"])
        scrollbar.pack(side="right", fill="y")

        self.op_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            bg=C["card"], fg=C["text"],
            selectbackground=C["accent"], selectforeground="#fff",
            font=("Segoe UI", 10), bd=0, highlightthickness=0,
            activestyle="none", relief="flat", cursor="hand2"
        )
        self.op_listbox.pack(fill="both", expand=True)
        scrollbar.config(command=self.op_listbox.yview)
        self.op_listbox.bind("<<ListboxSelect>>", self._on_op_select)

        # ── Parameter panel
        tk.Label(sidebar, text="PARAMETERS", font=("Segoe UI", 8, "bold"),
                 bg=C["sidebar"], fg=C["subtext"]).pack(anchor="w", padx=16, pady=(10, 2))

        self.param_frame = tk.Frame(sidebar, bg=C["sidebar"])
        self.param_frame.pack(fill="x", padx=8, pady=(0, 6))

        # ── Apply button
        apply_btn = tk.Button(
            sidebar, text="▶   Apply Operation",
            font=("Segoe UI", 11, "bold"),
            bg=C["accent"], fg="#fff",
            activebackground=C["accent2"], activeforeground="#fff",
            bd=0, padx=0, pady=12, cursor="hand2",
            command=self.apply_operation, relief="flat"
        )
        apply_btn.pack(fill="x", padx=8, pady=(2, 12))

        # Select first group
        self._select_group(groups[0])

    def _build_workspace(self, parent):
        """Right area: two image panels side by side."""
        workspace = tk.Frame(parent, bg=C["bg"])
        workspace.pack(side="left", fill="both", expand=True)

        panels = tk.Frame(workspace, bg=C["bg"])
        panels.pack(fill="both", expand=True)
        panels.columnconfigure(0, weight=1)
        panels.columnconfigure(1, weight=1)
        panels.rowconfigure(0, weight=1)

        self.orig_panel  = self._image_panel(panels, "Original Image",  0)
        self.res_panel   = self._image_panel(panels, "Result Image",     1)

        # Progress / info bar
        info_bar = tk.Frame(workspace, bg=C["panel"], height=32)
        info_bar.pack(fill="x")
        info_bar.pack_propagate(False)
        self.info_var = tk.StringVar(value="")
        tk.Label(info_bar, textvariable=self.info_var, font=("Segoe UI", 9),
                 bg=C["panel"], fg=C["subtext"]).pack(side="left", padx=12)
        self.progress = ttk.Progressbar(info_bar, mode="indeterminate", length=160)
        self.progress.pack(side="right", padx=12, pady=6)

    def _image_panel(self, parent, title, col):
        """Create a labelled image display panel."""
        frame = tk.Frame(parent, bg=C["card"], bd=0, relief="flat")
        frame.grid(row=0, column=col, sticky="nsew",
                   padx=(0 if col else 0, 6 if col == 0 else 0), pady=0)

        # Header
        header = tk.Frame(frame, bg=C["border"], height=32)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(header, text=title, font=("Segoe UI", 9, "bold"),
                 bg=C["border"], fg=C["text"]).pack(side="left", padx=12, pady=5)
        info_lbl = tk.Label(header, text="", font=("Segoe UI", 8),
                            bg=C["border"], fg=C["subtext"])
        info_lbl.pack(side="right", padx=10)

        # Canvas
        canvas = tk.Label(frame, bg=C["card"], text="",
                          anchor="center")
        canvas.pack(fill="both", expand=True, padx=4, pady=4)

        # Placeholder
        placeholder = tk.Label(
            frame,
            text="Drop image here\nor use Open Image ↑",
            font=("Segoe UI", 11), bg=C["card"], fg=C["subtext"]
        )

        if col == 0:
            self._orig_canvas   = canvas
            self._orig_info     = info_lbl
            self._orig_placeholder = placeholder
        else:
            self._res_canvas    = canvas
            self._res_info      = info_lbl
            self._res_placeholder = placeholder

        placeholder.place(relx=0.5, rely=0.5, anchor="center")
        return frame

    # ── SIDEBAR LOGIC ─────────────────────────────────────────

    def _select_group(self, group: str):
        """Switch the active operation group and repopulate listbox."""
        self.current_group.set(group)
        # Update tab styles
        for g, btn in self._tab_buttons.items():
            if g == group:
                btn.configure(bg=C["accent"], fg="#fff")
            else:
                btn.configure(bg=C["border"], fg=C["subtext"])

        self.op_listbox.delete(0, tk.END)
        for op_name in OPERATIONS[group].keys():
            self.op_listbox.insert(tk.END, f"  {op_name}")
        if OPERATIONS[group]:
            self.op_listbox.selection_set(0)
            self._on_op_select(None)

    def _on_op_select(self, _event):
        """When user clicks an operation in the listbox, update params panel."""
        sel = self.op_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        group = self.current_group.get()
        op_name = list(OPERATIONS[group].keys())[idx]
        self.current_op.set(op_name)
        _, params = OPERATIONS[group][op_name]
        self._rebuild_param_widgets(params)

    def _rebuild_param_widgets(self, params: dict):
        """Dynamically build sliders/entries for each parameter."""
        for w in self.param_frame.winfo_children():
            w.destroy()
        self._param_vars.clear()
        self._param_widgets.clear()

        if not params:
            tk.Label(self.param_frame, text="No parameters for this operation.",
                     font=("Segoe UI", 9), bg=C["sidebar"], fg=C["subtext"],
                     wraplength=260).pack(anchor="w", padx=4, pady=6)
            return

        SLIDER_CONFIGS = {
            "ksize":       (3, 31, 1),
            "sigma":       (0.1, 5.0, 0.1),
            "d":           (0, 10, 2),
            "Q":           (-5.0, 5.0, 0.1),
            "cutoff":      (5, 100, 5),
            "thresh":      (0, 255, 1),
            "block_size":  (3, 51, 2),
            "C":           (-10, 20, 1),
            "classes":     (2, 6, 1),
            "k":           (2, 10, 1),
            "n_segments":  (10, 400, 10),
            "scale":       (10, 500, 10),
            "quality":     (1, 100, 1),
            "pad":         (5, 100, 5),
            "top":         (0, 100, 5),
            "bottom":      (0, 100, 5),
            "left":        (0, 100, 5),
            "right":       (0, 100, 5),
            "low":         (0, 255, 5),
            "high":        (0, 255, 5),
            "compactness": (1, 50, 1),
            "sigma_seg":   (0.1, 2.0, 0.1),
            "min_size":    (10, 200, 10),
        }

        for name, default in params.items():
            row = tk.Frame(self.param_frame, bg=C["sidebar"])
            row.pack(fill="x", pady=2)
            lbl_text = name.replace("_", " ").title()

            if isinstance(default, float):
                var = tk.DoubleVar(value=default)
                cfg = SLIDER_CONFIGS.get(name, (0.0, 10.0, 0.1))
                val_lbl = tk.Label(row, text=f"{default:.1f}", width=5,
                                   font=("Segoe UI", 8), bg=C["sidebar"], fg=C["accent"])
                val_lbl.pack(side="right")
                s = ttk.Scale(row, from_=cfg[0], to=cfg[1], variable=var,
                              orient="horizontal",
                              command=lambda v, vl=val_lbl, vr=var, f=True:
                                  vl.configure(text=f"{float(v):.1f}"))
                s.pack(side="right", fill="x", expand=True, padx=(0, 4))
            else:
                var = tk.IntVar(value=int(default))
                cfg = SLIDER_CONFIGS.get(name, (1, 255, 1))
                val_lbl = tk.Label(row, text=str(default), width=4,
                                   font=("Segoe UI", 8), bg=C["sidebar"], fg=C["accent"])
                val_lbl.pack(side="right")
                s = ttk.Scale(row, from_=cfg[0], to=cfg[1], variable=var,
                              orient="horizontal",
                              command=lambda v, vl=val_lbl, vr=var:
                                  vl.configure(text=str(int(float(v)))))
                s.pack(side="right", fill="x", expand=True, padx=(0, 4))

            tk.Label(row, text=lbl_text + ":", font=("Segoe UI", 8),
                     bg=C["sidebar"], fg=C["text"], width=14,
                     anchor="w").pack(side="left")
            self._param_vars[name] = var
            self._param_widgets[name] = s

    # ── CORE ACTIONS ─────────────────────────────────────────

    def load_image(self):
        path = utils.open_file_dialog()
        if not path:
            return
        try:
            self.original_image = utils.load_image(path)
            self.result_image   = None
            self._display_image(self.original_image, orig=True)
            self._clear_result()
            self.status_var.set(f"Loaded: {path.split('/')[-1]}  —  {utils.image_info(self.original_image)}")
            self._orig_placeholder.place_forget()
        except Exception as e:
            utils.show_error("Load Error", str(e))

    def apply_operation(self):
        if not utils.ensure_image(self.original_image):
            utils.show_error("No Image", "Please open an image first.")
            return
        op_name = self.current_op.get().strip()
        if not op_name:
            utils.show_error("No Operation", "Please select an operation from the list.")
            return

        group = self.current_group.get()
        func, default_params = OPERATIONS[group][op_name]

        # Collect parameter values
        kwargs = {}
        for name in default_params:
            if name in self._param_vars:
                raw = self._param_vars[name].get()
                if isinstance(default_params[name], float):
                    kwargs[name] = float(raw)
                else:
                    kwargs[name] = int(float(raw))
            else:
                kwargs[name] = default_params[name]

        # Run in background thread to keep UI responsive
        self.progress.start(12)
        self.info_var.set(f"Running: {op_name} …")
        self.update_idletasks()

        def worker():
            try:
                result = func(self.original_image.copy(), **kwargs)
                self.after(0, lambda: self._on_result(result, op_name))
            except Exception as exc:
                self.after(0, lambda: self._on_error(str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_result(self, result, op_name):
        self.progress.stop()
        if result is None:
            self._on_error("Operation returned None.")
            return
        # Ensure 3-channel for consistent display
        if result.ndim == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        self.result_image = result
        self._display_image(result, orig=False)
        self._res_placeholder.place_forget()
        self.info_var.set(f"✓  {op_name}  —  {utils.image_info(result)}")
        self.status_var.set(f"Done: {op_name}")

    def _on_error(self, msg):
        self.progress.stop()
        self.info_var.set(f"Error: {msg}")
        utils.show_error("Processing Error", msg)

    def save_result(self):
        if not utils.ensure_image(self.result_image):
            utils.show_error("Nothing to Save", "Apply an operation first.")
            return
        path = utils.save_file_dialog()
        if path:
            if utils.save_image(self.result_image, path):
                utils.show_info("Saved", f"Image saved to:\n{path}")
            else:
                utils.show_error("Save Error", "Could not save image.")

    def reset(self):
        self.result_image = None
        self._clear_result()
        self.info_var.set("")
        self.status_var.set("Reset — select an operation and apply.")

    # ── DISPLAY HELPERS ───────────────────────────────────────

    def _display_image(self, image: np.ndarray, orig: bool):
        ph = utils.cv2_to_tk(image, max_w=580, max_h=560)
        if orig:
            self._orig_canvas.configure(image=ph, text="")
            self._orig_canvas.image = ph     # keep reference
            self._orig_info.configure(text=utils.image_info(image))
        else:
            self._res_canvas.configure(image=ph, text="")
            self._res_canvas.image = ph
            self._res_info.configure(text=utils.image_info(image))

    def _clear_result(self):
        self._res_canvas.configure(image="", text="")
        self._res_canvas.image = None
        self._res_info.configure(text="")
        self._res_placeholder.place(relx=0.5, rely=0.5, anchor="center")

    # ── HELPERS ───────────────────────────────────────────────

    def _btn(self, parent, text, cmd, bg, side="left", padx=4):
        b = tk.Button(parent, text=text, command=cmd,
                      font=("Segoe UI", 9, "bold"),
                      bg=bg, fg="#fff",
                      activebackground=C["accent2"], activeforeground="#fff",
                      bd=0, padx=12, pady=6, cursor="hand2", relief="flat")
        b.pack(side=side, padx=padx, pady=8)
        return b

    def _apply_ttk_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TScale",
                        background=C["sidebar"],
                        troughcolor=C["slider_bg"],
                        sliderthickness=14,
                        sliderrelief="flat")
        style.configure("TProgressbar",
                        troughcolor=C["panel"],
                        background=C["accent"],
                        borderwidth=0)


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = ImageProcessingApp()
    app.mainloop()
