import cv2
import tkinter as tk

from PIL import Image, ImageTk
from tkinter import filedialog, ttk


class App(tk.Frame):

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.parent.title("CBIR with Wavelet features")

        self.frame = ttk.Frame(root, padding=(3, 3, 3, 12))
        self.btn_open = ttk.Button(
            self.frame, text="Open", command=self.open_query_image)
        self.canvas = tk.Canvas(self.frame)
        self.lbl_n_matches = tk.Label(self.frame, text="Number of matches")
        self.n_matches = tk.IntVar()
        self.ent_n_matches = tk.Entry(self.frame, textvariable=self.n_matches)
        self.btn_n_matches = ttk.Button(self.frame, text="Search")

        self.frame.grid(column=0, row=0)
        self.btn_open.grid(column=0, row=0, columnspan=3)
        self.canvas.grid(column=0, row=1, columnspan=3)
        self.lbl_n_matches.grid(column=0, row=2)
        self.ent_n_matches.grid(column=1, row=2)
        self.btn_n_matches.grid(column=2, row=2)

        self.query_image = None
        self.query_image_canvas = self.canvas.create_image(
            0, 0, image=None, anchor=tk.NW)

        self.canvas.bind("<Button-1>", self.open_query_image)

    def open_query_image(self, *args):
        filename = tk.filedialog.askopenfilename()
        if len(filename):
            width = self.canvas.winfo_width()
            height = self.canvas.winfo_height()
            image = cv2.imread(filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (width, height))
            image = Image.fromarray(image)
            self.query_image = ImageTk.PhotoImage(image=image)
            self.canvas.itemconfig(
                self.query_image_canvas, image=self.query_image)


if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.minsize(800, 600)
    root.mainloop()
