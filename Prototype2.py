import tkinter as tk
from tkinter import filedialog, ttk
from transformers import pipeline

class SLMApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SLM Prototype UI")
        self.geometry("600x400")
        self.filepath = None

        # Load summarization pipeline
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
        self.filepath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
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
        
        btn = tk.Button(self, text="Process", command=self.process_file)
        btn.pack(pady=10)
    
    def process_file(self):
        option = self.selected_option.get()
        
        if option == "Summarize":
            self.create_output_page(self.summarize_file())
        else:
            self.create_output_page(f"The '{option}' option is not implemented yet.")
    
    def summarize_file(self):
        try:
            # Read file content
            with open(self.filepath, "r", encoding="utf-8") as file:
                text = file.read()
            
            # Limit the text length for the summarizer (it handles a maximum of ~1024 tokens)
            if len(text) > 3000:
                text = text[:3000]  # Truncate if the file is too large
            
            # Generate summary
            summary = self.summarizer(text, max_length=130, min_length=30, do_sample=False)
            return summary[0]['summary_text']
        
        except Exception as e:
            return f"An error occurred: {e}"
    
    def create_output_page(self, output):
        for widget in self.winfo_children():
            widget.destroy()
        
        label = tk.Label(self, text="Processing Complete!", font=("Arial", 14))
        label.pack(pady=20)
        
        output_text = tk.Text(self, height=15, width=60, wrap=tk.WORD)
        output_text.pack(pady=10)
        output_text.insert(tk.END, output)
        
        btn = tk.Button(self, text="Back to Home", command=self.create_front_page)
        btn.pack(pady=10)

if __name__ == "__main__":
    app = SLMApp()
    app.mainloop()
