import tkinter as tk


class UserActionFrame(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.__app = app

    def set_frame(self, frame):
        self.__subframe_obj = frame
        self.__subframe_obj.pack_forget()
        self.__subframe_obj.pack(fill="both", expand=True)
        self.__create_widgets()

    def __create_widgets(self):
        main_area = tk.Frame(self, borderwidth=0, relief="groove")
        main_area.pack(side="top", expand=True)

        self.__subframe_obj.pack(fill="both", expand=True, anchor="center")

        bottom_area = tk.Frame(self)
        bottom_area.pack(side="bottom", fill="x", padx=10, pady=10)

        back_button = tk.Button(bottom_area, text="Back", command=lambda: print('back'))
        back_button.pack(side="left")

        next_button = tk.Button(bottom_area, text="Next", command=lambda: print('next'))
        next_button.pack(side="right")
