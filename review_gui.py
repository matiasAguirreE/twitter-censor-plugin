# review_gui.py
"""
An interactive GUI tool to manually review and re-classify tweets from a CSV file.

This script provides a faster, more user-friendly way to clean up and verify
datasets compared to the command-line version. Includes navigation, editing,
and improved deletion functionality.

Usage
-----
# Start a review session for a given CSV file
python review_gui.py data/labeled/tweets_generated_all.csv

# Resume a session from a specific tweet ID
python review_gui.py data/labeled/tweets_generated_all.csv --start-from 123
"""
from __future__ import annotations

import argparse
import sys
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox, ttk

import pandas as pd

# --- Configuration ---
LABEL_INFO = {
    "V": {"name": "Violencia", "desc": "Violento"},
    "H": {"name": "Homofobia", "desc": "Homofóbico"},
    "X": {"name": "Xenofobia", "desc": "Xenofóbico"},
}
LABEL_COLUMNS = [info["name"] for info in LABEL_INFO.values()] + ["Incensurable"]

# --- Helper Functions ---

def get_label_string_from_row(row: pd.Series) -> str:
    """Converts the binary flags from a row into a human-readable string."""
    if row.get("Incensurable", 0) == 1:
        return "clean"
    parts = [code for code, info in LABEL_INFO.items() if row.get(info["name"], 0) == 1]
    return "+".join(sorted(parts)) if parts else "Unlabeled"

def update_row_from_label_string(label_str: str) -> dict[str, int]:
    """Creates a dictionary of new binary flags from a label string."""
    label_str = label_str.lower()
    new_labels = {col: 0 for col in LABEL_COLUMNS}
    if label_str in {"clean", "none", "incensurable"}:
        new_labels["Incensurable"] = 1
        return new_labels
    parts = {part.upper() for part in label_str.replace(" ", "").split("+")}
    for code in parts:
        if code in LABEL_INFO:
            new_labels[LABEL_INFO[code]["name"]] = 1
    return new_labels

# --- Main Application Class ---

class ReviewApp:
    def __init__(self, root: tk.Tk, filepath: str, start_id: int | None = None):
        self.root = root
        self.filepath = filepath
        self.df = self._load_data()
        self.reviewed_in_session = 0
        self.is_editing = False

        start_index = 0
        if start_id is not None:
            matching_indices = self.df.index[self.df['ID'] == start_id].tolist()
            if matching_indices:
                start_index = matching_indices[0]
            else:
                messagebox.showwarning("ID Not Found", f"Tweet with ID {start_id} not found.\nStarting from the beginning.")
        
        self.current_index = start_index

        self._setup_styles()
        self._setup_ui()
        
        self.root.protocol("WM_DELETE_WINDOW", self._save_and_exit)
        self._load_tweet_at_index(self.current_index)

    def _load_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.filepath)
            if 'ID' not in df.columns:
                raise ValueError("CSV must have an 'ID' column.")
            if '_to_delete' not in df.columns:
                df['_to_delete'] = False
            for col in LABEL_COLUMNS:
                if col not in df.columns:
                    df[col] = 0
            return df
        except FileNotFoundError:
            messagebox.showerror("Error", f"File not found:\n{self.filepath}")
            sys.exit(1)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read CSV:\n{e}")
            sys.exit(1)

    def _setup_styles(self):
        self.root.title("Tweet Review Tool")
        self.root.geometry("800x650")
        self.root.configure(bg="#282c34")

        style = ttk.Style()
        style.theme_use("clam")
        
        style.configure("TFrame", background="#282c34")
        style.configure("TLabel", background="#282c34", foreground="#abb2bf", font=("Helvetica", 12))
        style.configure("TButton", font=("Helvetica", 10, "bold"), borderwidth=0, padding=6)
        style.map("TButton",
                  background=[("active", "#61afef")],
                  foreground=[("active", "white")])

        style.configure("Header.TLabel", font=("Helvetica", 16, "bold"), foreground="#61afef")
        self.tweet_font = tkfont.Font(family="Helvetica", size=14)
        style.configure("Status.TLabel", font=("Helvetica", 11, "italic"))
        style.configure("Action.TButton", foreground="white", background="#5c6370")
        style.configure("Delete.TButton", foreground="white", background="#e06c75")
        style.configure("Label.TButton", foreground="white", background="#98c379")
        style.configure("Navigate.TButton", foreground="white", background="#61afef")
        style.configure("Edit.TButton", foreground="white", background="#c678dd")

    def _setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(expand=True, fill="both")

        self.progress_label = ttk.Label(main_frame, text="Loading...", style="Header.TLabel")
        self.progress_label.pack(pady=(0, 10))

        # --- Tweet Display ---
        self.tweet_frame = ttk.Frame(main_frame, relief="solid", borderwidth=1)
        self.tweet_frame.pack(expand=True, fill="both", pady=10)
        
        self.tweet_text_widget = tk.Text(
            self.tweet_frame, wrap="word", font=self.tweet_font,
            bg="#21252b", fg="white", insertbackground="white",
            borderwidth=0, highlightthickness=0, state="disabled",
            padx=10, pady=10
        )
        self.tweet_text_widget.pack(expand=True, fill="both")

        self.status_label = ttk.Label(main_frame, text="", style="Status.TLabel")
        self.status_label.pack(pady=(5, 10))

        # --- Classification Buttons ---
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(pady=10)
        label_combos = self._get_label_combos()
        max_cols = 4
        self.label_buttons = []
        for i, combo in enumerate(label_combos):
            row, col = divmod(i, max_cols)
            btn = ttk.Button(
                buttons_frame, text=combo, style="Label.TButton",
                command=lambda c=combo: self._classify_and_next(c)
            )
            btn.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
            self.label_buttons.append(btn)

        # --- Action Buttons ---
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(pady=(20, 0), fill="x")
        action_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.prev_button = ttk.Button(action_frame, text="Anterior", command=self._load_previous_tweet, style="Navigate.TButton")
        self.prev_button.grid(row=0, column=0, padx=5, sticky="ew")
        
        self.skip_button = ttk.Button(action_frame, text="Saltar", command=self._load_next_tweet, style="Action.TButton")
        self.skip_button.grid(row=0, column=1, padx=5, sticky="ew")

        self.edit_button = ttk.Button(action_frame, text="Editar", command=self._toggle_edit_mode, style="Edit.TButton")
        self.edit_button.grid(row=0, column=2, padx=5, sticky="ew")

        self.delete_button = ttk.Button(action_frame, text="Eliminar", command=self._delete_and_next, style="Delete.TButton")
        self.delete_button.grid(row=0, column=3, padx=5, sticky="ew")
        
        ttk.Button(main_frame, text="Guardar y Salir", command=self._save_and_exit, style="Action.TButton").pack(pady=(10,0), fill="x")

    def _get_label_combos(self) -> list[str]:
        keys = sorted(LABEL_INFO.keys())
        combos = list(keys)
        from itertools import combinations
        for i in range(2, len(keys) + 1):
            for combo_tuple in combinations(keys, i):
                combos.append("+".join(combo_tuple))
        combos.append("clean")
        return combos

    def _load_tweet_at_index(self, index: int):
        if not (0 <= index < len(self.df)):
            self.prev_button.config(state="disabled")
            return

        self.current_index = index
        row = self.df.iloc[self.current_index]
        
        self.progress_label.config(text=f"Tweet {self.current_index + 1} of {len(self.df)} (ID: {row['ID']})")
        
        self.tweet_text_widget.config(state="normal")
        self.tweet_text_widget.delete("1.0", tk.END)
        self.tweet_text_widget.insert("1.0", row["Tweet"])
        self.tweet_text_widget.config(state="disabled")

        current_label_str = get_label_string_from_row(row)
        is_deleted = row.get('_to_delete', False)
        
        status_text = f"Etiquetas Actuales: {current_label_str}"
        if is_deleted:
            status_text += "  -- [MARCADO PARA ELIMINAR]"
            self.status_label.config(foreground="#e06c75")
        else:
            self.status_label.config(foreground="#abb2bf")
        self.status_label.config(text=status_text)

        self.prev_button.config(state="normal" if self.current_index > 0 else "disabled")
        self.skip_button.config(state="normal")

    def _load_next_tweet(self):
        if self.current_index + 1 >= len(self.df):
            self._show_completion_and_exit()
        else:
            self._load_tweet_at_index(self.current_index + 1)

    def _load_previous_tweet(self):
        if self.current_index > 0:
            self._load_tweet_at_index(self.current_index - 1)

    def _classify_and_next(self, label_str: str):
        new_labels = update_row_from_label_string(label_str)
        for col, value in new_labels.items():
            self.df.loc[self.current_index, col] = value
        
        self.df.loc[self.current_index, '_to_delete'] = False
        self.reviewed_in_session += 1
        self._load_next_tweet()

    def _delete_and_next(self):
        self.df.loc[self.current_index, '_to_delete'] = True
        self.reviewed_in_session += 1
        self._load_next_tweet()

    def _toggle_edit_mode(self):
        self.is_editing = not self.is_editing
        if self.is_editing:
            self.tweet_text_widget.config(state="normal")
            self.edit_button.config(text="Guardar Cambio")
            for btn in self.label_buttons + [self.prev_button, self.skip_button, self.delete_button]:
                btn.config(state="disabled")
        else: # Saving
            new_text = self.tweet_text_widget.get("1.0", tk.END).strip()
            self.df.loc[self.current_index, "Tweet"] = new_text
            self.tweet_text_widget.config(state="disabled")
            self.edit_button.config(text="Editar")
            for btn in self.label_buttons + [self.skip_button, self.delete_button]:
                btn.config(state="normal")
            self.prev_button.config(state="normal" if self.current_index > 0 else "disabled")
            self.reviewed_in_session += 1

    def _show_completion_and_exit(self):
        if messagebox.askyesno("Completado", "Has revisado todos los tweets. ¿Quieres guardar y salir?"):
            self._save_and_exit()
        else:
            self._load_tweet_at_index(self.current_index)

    def _save_and_exit(self):
        print(f"\nGuardando cambios en {self.filepath}...")
        try:
            df_to_save = self.df[self.df['_to_delete'] == False].copy()
            df_to_save.drop(columns=['_to_delete'], inplace=True)
            df_to_save.to_csv(self.filepath, index=False)
            print(f"¡Éxito! Se revisaron {self.reviewed_in_session} tweets en esta sesión.")
        except Exception as e:
            print(f"Error al guardar el archivo: {e}", file=sys.stderr)
            messagebox.showerror("Error al Guardar", f"No se pudo guardar el archivo.\n{e}")
        
        self.root.destroy()

def main():
    parser = argparse.ArgumentParser(
        description="Herramienta GUI para revisar y clasificar tweets.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("filepath", help="Ruta al archivo CSV a revisar.")
    parser.add_argument(
        "--start-from",
        type=int,
        default=None,
        help="ID del tweet desde el cual comenzar la revisión (para reanudar sesiones)."
    )
    args = parser.parse_args()

    root = tk.Tk()
    app = ReviewApp(root, args.filepath, start_id=args.start_from)
    root.mainloop()

if __name__ == "__main__":
    main()
