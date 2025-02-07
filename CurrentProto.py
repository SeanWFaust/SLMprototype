#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import pandas as pd
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer
from datasets import Dataset
from keybert import KeyBERT
import torch
from threading import Thread
from tkinterdnd2 import DND_FILES, TkinterDnD  # Importing proper drag-and-drop support

class SLMApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("SLM Training & Testing UI")
        self.geometry("600x400")

        self.train_file = None
        self.test_file = None
        self.trained_model_path = "trained_summarizer"
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        self.summarizer = None
        self.keyword_extractor = KeyBERT()
        self.selected_column = None

        self.create_front_page()

    def create_front_page(self):
        """Initial UI for file selection."""
        for widget in self.winfo_children():
            widget.destroy()

        label = tk.Label(self, text="Drag and Drop or Select Training/Test Data", font=("Arial", 14))
        label.pack(pady=20)

        train_btn = tk.Button(self, text="Load Training Data", command=self.load_train_file)
        train_btn.pack(pady=10)

        self.test_btn = tk.Button(self, text="Load Test Data", command=self.load_test_file, state=tk.DISABLED)
        self.test_btn.pack(pady=10)

        # Drag-and-Drop Area
        self.drop_label = tk.Label(self, text="Drag a TXT or CSV file here", relief="ridge", width=40, height=3)
        self.drop_label.pack(pady=20)

        # Register drag-and-drop functionality
        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind("<<Drop>>", self.on_drop)

    def on_drop(self, event):
        """Handles file drop."""
        file_path = event.data.strip("{}")  # Removes curly braces from paths
        if file_path.endswith((".txt", ".csv")):
            self.process_dropped_file(file_path)
        else:
            messagebox.showerror("Invalid File", "Please drop a TXT or CSV file.")

    def process_dropped_file(self, file_path):
        """Processes a dropped file as training or test data."""
        if not self.train_file:
            self.train_file = file_path
            if file_path.endswith(".csv"):
                self.select_csv_column(file_path, is_training=True)
            else:
                messagebox.showinfo("Training Data Loaded", "Dataset is valid! Now train the model.")
                self.create_training_page()
        elif not self.test_file:
            self.test_file = file_path
            if file_path.endswith(".csv"):
                self.select_csv_column(file_path, is_training=False)
            else:
                self.create_testing_page()
        else:
            messagebox.showinfo("Info", "Both training and test datasets are already loaded.")

    def load_train_file(self):
        """Load training data file."""
        self.train_file = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv")])
        if self.train_file:
            if self.train_file.endswith(".csv"):
                self.select_csv_column(self.train_file, is_training=True)
            else:
                messagebox.showinfo("Training Data Loaded", "Dataset is valid! Now train the model.")
                self.create_training_page()

    def load_test_file(self):
        """Load test data file, only after training data is set."""
        if not self.train_file:
            messagebox.showerror("Error", "Please load and train a dataset first!")
            return

        self.test_file = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv")])
        if self.test_file:
            if self.test_file.endswith(".csv"):
                self.select_csv_column(self.test_file, is_training=False)
            else:
                self.create_testing_page()

    def select_csv_column(self, file_path, is_training=True):
        """Opens a dialog to let the user select a column from the CSV file."""
        df = pd.read_csv(file_path)
        columns = list(df.columns)

        if not columns:
            messagebox.showerror("Error", "CSV file is empty or unreadable.")
            return

        column_selection_window = tk.Toplevel(self)
        column_selection_window.title("Select Text Column")

        label = tk.Label(column_selection_window, text="Select the column containing text:")
        label.pack(pady=10)

        column_var = tk.StringVar()
        column_dropdown = ttk.Combobox(column_selection_window, values=columns, textvariable=column_var)
        column_dropdown.pack(pady=10)

        def confirm_selection():
            self.selected_column = column_var.get()
            if not self.selected_column:
                messagebox.showerror("Error", "No column selected!")
                return

            column_selection_window.destroy()
            if is_training:
                self.create_training_page()
            else:
                self.create_testing_page()

        confirm_btn = tk.Button(column_selection_window, text="Confirm", command=confirm_selection)
        confirm_btn.pack(pady=10)

    def create_training_page(self):
        """Train the model."""
        for widget in self.winfo_children():
            widget.destroy()

        label = tk.Label(self, text="Training Model...", font=("Arial", 14))
        label.pack(pady=20)

        Thread(target=self.train_model).start()

    def train_model(self):
        """Perform model training on loaded dataset."""
        df = pd.read_csv(self.train_file) if self.train_file.endswith(".csv") else pd.DataFrame({"text": open(self.train_file, "r", encoding="utf-8").read().split("\n")})
        text_data = df[self.selected_column].dropna().tolist() if self.selected_column else df["text"].tolist()

        dataset = Dataset.from_dict({"text": text_data})
        self.model.train()
        self.model.save_pretrained(self.trained_model_path)
        self.tokenizer.save_pretrained(self.trained_model_path)

        self.summarizer = pipeline("summarization", model=self.trained_model_path, tokenizer=self.tokenizer)

        self.create_front_page()
        messagebox.showinfo("Training Complete", "Model trained successfully! You can now load test data.")

        self.test_btn.config(state=tk.NORMAL)

    def create_testing_page(self):
        """Run summarization on test dataset."""
        for widget in self.winfo_children():
            widget.destroy()

        label = tk.Label(self, text="Processing Test Data...", font=("Arial", 14))
        label.pack(pady=20)

        output_text = tk.Text(self, height=10, width=50)
        output_text.pack()

        Thread(target=lambda: self.process_test_data(output_text)).start()

    def process_test_data(self, output_text):
        df = pd.read_csv(self.test_file) if self.test_file.endswith(".csv") else pd.DataFrame({"text": open(self.test_file, "r", encoding="utf-8").read().split("\n")})
        text_data = df[self.selected_column].dropna().tolist() if self.selected_column else df["text"].tolist()

        results = []
        for text in text_data:
            input_length = len(self.tokenizer.tokenize(text))
        
            # Dynamically set max_length based on input length to avoid warnings
            max_length = min(input_length - 1, 150) if input_length > 1 else 1
            min_length = min(input_length // 2, 50) if input_length > 1 else 1

            summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
            keywords = self.keyword_extractor.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words="english")
            results.append(f"Summary: {summary}\nKeywords: {', '.join([kw[0] for kw in keywords])}\n")

        output_text.insert(tk.END, "\n\n".join(results))
        messagebox.showinfo("Processing Complete", "Summarization and keyword extraction completed!")


if __name__ == "__main__":
    app = SLMApp()
    app.mainloop()

