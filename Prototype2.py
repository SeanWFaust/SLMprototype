#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter as tk
from tkinter import filedialog, ttk

class SLMApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SLM Prototype UI")
        self.geometry("600x400")
        self.filepath = None
        
        self.create_front_page()
    
    def create_front_page(self):
        for widget in self.winfo_children():
            widget.destroy()
        
        label = tk.Label(self, text="Drag and Drop Your File Here", font=("Arial", 14))
        label.pack(pady=20)
        
        btn = tk.Button(self, text="Browse File", command=self.load_file)
        btn.pack(pady=10)
    
    def load_file(self):
        self.filepath = filedialog.askopenfilename()
        if self.filepath:
            self.create_selection_page()
    
    def create_selection_page(self):
        for widget in self.winfo_children():
            widget.destroy()
        
        label = tk.Label(self, text="Select Processing Option", font=("Arial", 14))
        label.pack(pady=20)
        
        options = ["Summarize", "Classify", "Extract Keywords"]
        self.selected_option = tk.StringVar()
        self.selected_option.set(options[0])
        
        dropdown = ttk.Combobox(self, values=options, textvariable=self.selected_option)
        dropdown.pack(pady=10)
        
        btn = tk.Button(self, text="Process", command=self.create_output_page)
        btn.pack(pady=10)
    
    def create_output_page(self):
        for widget in self.winfo_children():
            widget.destroy()
        
        label = tk.Label(self, text="Processing Complete!", font=("Arial", 14))
        label.pack(pady=20)
        
        output_text = tk.Text(self, height=10, width=50)
        output_text.pack()
        output_text.insert(tk.END, f"Processed data from: {self.filepath}\nUsing: {self.selected_option.get()}")
        
        btn = tk.Button(self, text="Back to Home", command=self.create_front_page)
        btn.pack(pady=10)

if __name__ == "__main__":
    app = SLMApp()
    app.mainloop()


# In[2]:


import tkinter as tk
from tkinter import filedialog, ttk
from transformers import pipeline

class SLMApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SLM Prototype UI")
        self.geometry("600x400")
        self.filepath = None
        self.summarizer = pipeline("summarization")
        
        self.create_front_page()
    
    def create_front_page(self):
        for widget in self.winfo_children():
            widget.destroy()
        
        label = tk.Label(self, text="Drag and Drop Your File Here", font=("Arial", 14))
        label.pack(pady=20)
        
        btn = tk.Button(self, text="Browse File", command=self.load_file)
        btn.pack(pady=10)
    
    def load_file(self):
        self.filepath = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if self.filepath:
            self.create_selection_page()
    
    def create_selection_page(self):
        for widget in self.winfo_children():
            widget.destroy()
        
        label = tk.Label(self, text="Select Processing Option", font=("Arial", 14))
        label.pack(pady=20)
        
        options = ["Summarize"]
        self.selected_option = tk.StringVar()
        self.selected_option.set(options[0])
        
        dropdown = ttk.Combobox(self, values=options, textvariable=self.selected_option)
        dropdown.pack(pady=10)
        
        btn = tk.Button(self, text="Process", command=self.create_output_page)
        btn.pack(pady=10)
    
    def create_output_page(self):
        for widget in self.winfo_children():
            widget.destroy()
        
        label = tk.Label(self, text="Processing Complete!", font=("Arial", 14))
        label.pack(pady=20)
        
        output_text = tk.Text(self, height=10, width=50)
        output_text.pack()
        
        summary = self.process_text()
        output_text.insert(tk.END, summary)
        
        btn = tk.Button(self, text="Back to Home", command=self.create_front_page)
        btn.pack(pady=10)
    
    def process_text(self):
        if not self.filepath:
            return "No file selected."
        
        with open(self.filepath, "r", encoding="utf-8") as file:
            text = file.read()
        
        if len(text) > 1024:
            text = text[:1024]  # Limit input size for the model
        
        summary = self.summarizer(text, max_length=150, min_length=50, do_sample=False)
        return summary[0]['summary_text']

if __name__ == "__main__":
    app = SLMApp()
    app.mainloop()


# In[2]:


import tkinter as tk
from tkinter import filedialog, ttk
from transformers import pipeline

class SLMApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SLM Prototype UI")
        self.geometry("600x400")
        self.filepath = None
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        self.create_front_page()
    
    def create_front_page(self):
        for widget in self.winfo_children():
            widget.destroy()
        
        label = tk.Label(self, text="Drag and Drop Your File Here", font=("Arial", 14))
        label.pack(pady=20)
        
        btn = tk.Button(self, text="Browse File", command=self.load_file)
        btn.pack(pady=10)
    
    def load_file(self):
        self.filepath = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if self.filepath:
            self.create_selection_page()
    
    def create_selection_page(self):
        for widget in self.winfo_children():
            widget.destroy()
        
        label = tk.Label(self, text="Select Processing Option", font=("Arial", 14))
        label.pack(pady=20)
        
        options = ["Summarize"]
        self.selected_option = tk.StringVar()
        self.selected_option.set(options[0])
        
        dropdown = ttk.Combobox(self, values=options, textvariable=self.selected_option)
        dropdown.pack(pady=10)
        
        btn = tk.Button(self, text="Process", command=self.create_output_page)
        btn.pack(pady=10)
    
    def create_output_page(self):
        for widget in self.winfo_children():
            widget.destroy()
        
        label = tk.Label(self, text="Processing Complete!", font=("Arial", 14))
        label.pack(pady=20)
        
        output_text = tk.Text(self, height=10, width=50)
        output_text.pack()
        
        summary = self.process_text()
        output_text.insert(tk.END, summary)
        
        btn = tk.Button(self, text="Back to Home", command=self.create_front_page)
        btn.pack(pady=10)
    
    def process_text(self):
        if not self.filepath:
            return "No file selected."
        
        with open(self.filepath, "r", encoding="utf-8") as file:
            text = file.read()
        
        # Process text in chunks if needed to avoid length restrictions
        chunk_size = 1024
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        summaries = [self.summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text'] for chunk in chunks]
        
        return " ".join(summaries)

if __name__ == "__main__":
    app = SLMApp()
    app.mainloop()


# In[1]:


pip install torch transformers datasets keybert


# In[1]:


import tkinter as tk
from tkinter import filedialog, ttk
from transformers import pipeline
from keybert import KeyBERT

class SLMApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SLM Prototype UI")
        self.geometry("600x400")
        self.filepath = None
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.keyword_extractor = KeyBERT()

        self.create_front_page()

    def create_front_page(self):
        """Initial UI for file selection."""
        for widget in self.winfo_children():
            widget.destroy()
        
        label = tk.Label(self, text="Drag and Drop Your File Here", font=("Arial", 14))
        label.pack(pady=20)
        
        btn = tk.Button(self, text="Browse File", command=self.load_file)
        btn.pack(pady=10)
    
    def load_file(self):
        """Handles file selection."""
        self.filepath = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if self.filepath:
            self.create_selection_page()
    
    def create_selection_page(self):
        """UI for selecting processing options."""
        for widget in self.winfo_children():
            widget.destroy()
        
        label = tk.Label(self, text="Select Processing Option", font=("Arial", 14))
        label.pack(pady=20)
        
        options = ["Summarize", "Extract Keywords"]
        self.selected_option = tk.StringVar()
        self.selected_option.set(options[0])
        
        dropdown = ttk.Combobox(self, values=options, textvariable=self.selected_option)
        dropdown.pack(pady=10)
        
        btn = tk.Button(self, text="Process", command=self.create_output_page)
        btn.pack(pady=10)
    
    def create_output_page(self):
        """Displays the output after processing."""
        for widget in self.winfo_children():
            widget.destroy()
        
        label = tk.Label(self, text="Processing Complete!", font=("Arial", 14))
        label.pack(pady=20)
        
        output_text = tk.Text(self, height=10, width=60)
        output_text.pack()
        
        result = self.process_text()
        output_text.insert(tk.END, result)
        
        btn = tk.Button(self, text="Back to Home", command=self.create_front_page)
        btn.pack(pady=10)
    
    def process_text(self):
        """Processes the text based on user selection."""
        if not self.filepath:
            return "No file selected."

        with open(self.filepath, "r", encoding="utf-8") as file:
            text = file.read()

        # Summarization
        if self.selected_option.get() == "Summarize":
            chunk_size = 1024  # Hugging Face models have a token limit
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            summaries = [self.summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text'] for chunk in chunks]
            return " ".join(summaries)

        # Keyword Extraction
        elif self.selected_option.get() == "Extract Keywords":
            keywords = self.keyword_extractor.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=10)
            return "Top Keywords:\n" + ", ".join([kw[0] for kw in keywords])

if __name__ == "__main__":
    app = SLMApp()
    app.mainloop()


# In[3]:


import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
from keybert import KeyBERT
import torch
from threading import Thread

class SLMApp(tk.Tk):
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

        self.create_front_page()

    def create_front_page(self):
        """Initial UI for file selection."""
        for widget in self.winfo_children():
            widget.destroy()

        label = tk.Label(self, text="Select Training or Test Data", font=("Arial", 14))
        label.pack(pady=20)

        train_btn = tk.Button(self, text="Load Training Data", command=self.load_train_file)
        train_btn.pack(pady=10)

        self.test_btn = tk.Button(self, text="Load Test Data", command=self.load_test_file, state=tk.DISABLED)
        self.test_btn.pack(pady=10)

    def load_train_file(self):
        """Load training data file."""
        self.train_file = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if self.train_file:
            if self.validate_dataset():
                messagebox.showinfo("Training Data Loaded", "Dataset is valid! Now train the model.")
                self.create_training_page()
            else:
                self.train_file = None  # Reset if invalid

    def validate_dataset(self):
        """Validates training dataset for minimum quality."""
        with open(self.train_file, "r", encoding="utf-8") as file:
            lines = [line.strip() for line in file.readlines() if line.strip()]

        if len(lines) < 10:
            messagebox.showerror("Error", "Dataset is too small! At least 10 lines required.")
            return False

        if any(len(line) < 50 for line in lines):
            messagebox.showwarning("Warning", "Some lines are very short. This may affect training quality.")

        return True

    def create_training_page(self):
        """UI for training the model."""
        for widget in self.winfo_children():
            widget.destroy()

        label = tk.Label(self, text="Train Your Model", font=("Arial", 14))
        label.pack(pady=20)

        self.progress = ttk.Progressbar(self, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(pady=10)

        self.status_label = tk.Label(self, text="Ready to train.", font=("Arial", 10))
        self.status_label.pack(pady=10)

        train_btn = tk.Button(self, text="Start Training", command=self.start_training)
        train_btn.pack(pady=10)

    def start_training(self):
        """Runs training in a separate thread."""
        self.progress["value"] = 0
        self.status_label.config(text="Training in progress...")
        Thread(target=self.train_model).start()

    def train_model(self):
        """Fine-tune the summarization model using training data."""
        if not self.train_file:
            return

        with open(self.train_file, "r", encoding="utf-8") as file:
            text = file.read().split("\n")

        dataset = Dataset.from_dict({"text": text})

        def preprocess_function(examples):
            inputs = [doc for doc in examples["text"]]
            model_inputs = self.tokenizer(inputs, max_length=1024, truncation=True)
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(inputs, max_length=150, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_datasets = dataset.map(preprocess_function, batched=True)
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)

        training_args = TrainingArguments(
            output_dir=self.trained_model_path,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=3,
            weight_decay=0.01,
            save_total_limit=1,
            logging_dir="./logs",
            logging_steps=10
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )

        for epoch in range(1, 4):
            trainer.train()
            self.progress["value"] += 33  # 3 epochs, so increment by ~33% per epoch
            self.update_idletasks()

        self.model.save_pretrained(self.trained_model_path)
        self.tokenizer.save_pretrained(self.trained_model_path)

        self.summarizer = pipeline("summarization", model=self.trained_model_path)
        messagebox.showinfo("Training Complete", "Model training finished!")

        self.status_label.config(text="Training complete! You may now test your model.")
        self.progress["value"] = 100
        self.test_btn.config(state=tk.NORMAL)  # Enable test data selection
        self.create_front_page()

    def load_test_file(self):
        """Load test data for summarization or keyword extraction."""
        self.test_file = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if self.test_file:
            self.create_test_page()

    def create_test_page(self):
        """UI for selecting processing options on test data."""
        for widget in self.winfo_children():
            widget.destroy()

        label = tk.Label(self, text="Select Processing Option", font=("Arial", 14))
        label.pack(pady=20)

        options = ["Summarize", "Extract Keywords"]
        self.selected_option = tk.StringVar()
        self.selected_option.set(options[0])

        dropdown = ttk.Combobox(self, values=options, textvariable=self.selected_option)
        dropdown.pack(pady=10)

        btn = tk.Button(self, text="Process", command=self.create_output_page)
        btn.pack(pady=10)

    def create_output_page(self):
        """Displays the output after processing."""
        for widget in self.winfo_children():
            widget.destroy()

        label = tk.Label(self, text="Processing Complete!", font=("Arial", 14))
        label.pack(pady=20)

        output_text = tk.Text(self, height=10, width=60)
        output_text.pack()

        result = self.process_test_text()
        output_text.insert(tk.END, result)

        btn = tk.Button(self, text="Back to Home", command=self.create_front_page)
        btn.pack(pady=10)

    def process_test_text(self):
        """Processes the test data based on user selection."""
        if not self.test_file:
            return "No test data selected."

        with open(self.test_file, "r", encoding="utf-8") as file:
            text = file.read()

        if self.selected_option.get() == "Summarize":
            chunks = [text[i:i+1024] for i in range(0, len(text), 1024)]
            summaries = [self.summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text'] for chunk in chunks]
            return " ".join(summaries)

        elif self.selected_option.get() == "Extract Keywords":
            keywords = self.keyword_extractor.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=10)
            return "Top Keywords:\n" + ", ".join([kw[0] for kw in keywords])

if __name__ == "__main__":
    app = SLMApp()
    app.mainloop()


# In[6]:


pip install tkinterdnd2 pandas torch transformers keybert


# In[1]:


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
        """Process test data for summarization and keyword extraction."""
        df = pd.read_csv(self.test_file) if self.test_file.endswith(".csv") else pd.DataFrame({"text": open(self.test_file, "r", encoding="utf-8").read().split("\n")})
        text_data = df[self.selected_column].dropna().tolist() if self.selected_column else df["text"].tolist()

        results = []
        for text in text_data:
            summary = self.summarizer(text[:1024], max_length=150, min_length=50, do_sample=False)[0]['summary_text']
            keywords = self.keyword_extractor.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words="english")
            results.append(f"Summary: {summary}\nKeywords: {', '.join([kw[0] for kw in keywords])}\n")

        output_text.insert(tk.END, "\n\n".join(results))
        messagebox.showinfo("Processing Complete", "Summarization and keyword extraction completed!")

if __name__ == "__main__":
    app = SLMApp()
    app.mainloop()


# In[ ]:




