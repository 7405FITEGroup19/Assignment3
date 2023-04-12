# -*- coding: utf-8 -*-
"""User_interface.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13__rxzercTBjgULx1zNXyXK61fWX5KRM
"""

"""
A graphical user interface for users to easily price various options with your pricer.
"""

from American_option_pricer import american_option
from Arithmetic_Asian_option_pricer import arithmetic_asia_option
from Arithmetic_basket_option_pricer import arithmetic_bascket_option
from European_Option_Black_Scholes_Formulas import black_scholes
from Geometric_Asian_option_pricer import geometric_asia_option
from Geometric_basket_option_pricer import geometric_bascket_option
from Implied_volatility import vega, Newton_Raphson4Ivolatility
from KIKO_Put_Option_Quasi_Monte_Carlo import Quasi_Monte_Carlo

from tkinter import *
from tkinter import ttk, scrolledtext
import tkinter.font as tkFont
import math



class OptionPricerGUI:

    def __init__(self):
        self.create_main_window()
        self.create_frames()
        self.create_menu()

        self.show_home_page()

        self.window.mainloop()

    def create_main_window(self):
        self.window = Tk()
        self.window.title("Option Pricer")
        self.window.geometry('720x460')
        self.menu_bar = Menu(self.window)

    def create_frames(self):
        self.frames = {
            'home': Frame(self.window),
            'pricer1': Frame(self.window),
            'pricer2': Frame(self.window),
            'pricer3': Frame(self.window),
            'pricer4': Frame(self.window),
            'pricer5': Frame(self.window),
            'pricer6': Frame(self.window),
            'pricer7': Frame(self.window),
            'pricer8': Frame(self.window),
        }

    def create_menu(self):
        self.create_homepage_menu()
        self.create_pricer_menu()
        self.window.config(menu=self.menu_bar)

    def create_homepage_menu(self):
        homepage_menu = Menu(self.menu_bar, tearoff=0)
        homepage_menu.add_command(label="Homepage", command=self.show_home_page)
        homepage_menu.add_command(label="Quit", command=self.quit_application)
        self.menu_bar.add_cascade(label='Homepage', menu=homepage_menu)

    def create_pricer_menu(self):
        pricer_menu = Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label='Select Pricer Model', menu=pricer_menu)

        pricer_menu.add_command(label="Pricer 1: European Options - Black-Scholes Formulas",
                                command=self.task1)
        pricer_menu.add_command(label="Pricer 2: Implied Volatility - European Options",
                                command=self.task2)
        pricer_menu.add_command(label="Pricer 3: Geometric Asian Options - Closed-Form Formulas",
                                command=self.task3)
        pricer_menu.add_command(label="Pricer 4: Geometric Basket Options - Closed-Form Formulas",
                                command=self.task4)
        pricer_menu.add_command(label="Pricer 5: Arithmetic Asian Options - Monte Carlo Method",
                                command=self.task5)
        pricer_menu.add_command(label="Pricer 6: Arithmetic Mean Basket Options - Monte Carlo Method",
                                command=self.task6)
        pricer_menu.add_command(label="Pricer 7: KIKO-put option - Quasi-Monte Carlo method",
                                command=self.task8)
        pricer_menu.add_command(label="Pricer 8: American Options - Binomial Tree Method",
                                command=self.task7)

    def show_home_page(self):
        self.hide_all_frames()
        self.frames['home'].pack()

        ft1 = tkFont.Font(size=29)
        Label(self.frames['home'], text="Option Pricer", font=ft1, fg="grey", height=11).pack()
        ft2 = tkFont.Font(size=12)
        Label(self.frames['home'], text="Authors: Wu Yufan, Zhang Jingya, Zhao Mingyang", font=ft2, fg="grey",
              height=11).pack()

    def quit_application(self):
        self.window.destroy()

    def hide_all_frames(self):
        for frame in self.frames.values():
            frame.pack_forget()

    def show_frame(self, frame_key):
        self.hide_all_frames()
        self.frames[frame_key].pack()

    # Black-Scholes Formulas for European call/put option.
    def task1(self):

        self.hide_all_frames()
        frame = self.frames['pricer1']
        frame.pack()

        Label(frame, text = " Black-Scholes Formulas for European Call/Put Options.", justify = "right").grid(row = 1, column = 1,sticky = W)
        Label(frame, text = "Spot Price of Asset:").grid(row = 2, column = 1, sticky = W)
        Label(frame, text = "Volatility:").grid(row = 3, column = 1, sticky = W)
        Label(frame, text = "Risk-free Interest Rate:").grid(row = 4, column = 1, sticky = W)
        Label(frame, text = "Repo Rate:").grid(row = 5, column = 1, sticky = W)
        Label(frame, text = "Time to Maturity (in years):").grid(row = 6, column = 1, sticky = W)
        Label(frame, text = "Strike:").grid(row = 7, column = 1, sticky = W)
        Label(frame, text = "Option Type:").grid(row = 8, column = 1, sticky = W)

        self.s0 = DoubleVar()
        self.s0.set(100)
        self.sigma = DoubleVar()
        self.sigma.set(0.2)
        self.r = DoubleVar()
        self.r.set(0.01)
        self.repo = DoubleVar()
        self.repo.set(0.0)
        self.T = DoubleVar()
        self.T.set(0.5)
        self.K = DoubleVar()
        self.K.set(100)
        self.option_type = StringVar()


        Entry(frame, textvariable = self.s0).grid(row = 2, column = 2, sticky = W)
        Entry(frame, textvariable = self.sigma).grid(row = 3, column = 2, sticky = W)
        Entry(frame, textvariable = self.r).grid(row = 4, column = 2, sticky = W)
        Entry(frame, textvariable = self.repo).grid(row = 5, column = 2, sticky = W)
        Entry(frame, textvariable = self.T).grid(row = 6, column = 2, sticky = W)
        Entry(frame, textvariable = self.K).grid(row = 7, column = 2, sticky = W)


        self.option_type_selector = ttk.Combobox(frame, width = 17, values = ("Select Option Type", "Call Option", "Put Option"), textvariable = self.option_type, postcommand = self.run_task1)
        self.option_type_selector.current(0)
        self.option_type_selector.grid(row = 8, column = 2, sticky = W)

        Button(frame, width = 10, text = "Reset", command = self.reset_task1).grid(row = 10, column = 2, columnspan = 1, sticky = E)


        Button(frame, width = 10, text = "Run", command = self.run_task1).grid(row = 10, column = 2, columnspan = 1, sticky = W)


        self.logs = scrolledtext.ScrolledText(frame, width = 74, height = 12)
        self.logs.grid(row = 11, column = 1, rowspan = 4, columnspan = 2, sticky = W)

    # Implied volatility
    def task2(self):

        self.hide_all_frames()
        frame = self.frames['pricer2']
        frame.pack()

        Label(frame, text = "Implied Volatility Calculator for European Options", justify = "right").grid(row = 1, column = 1,sticky = W)
        Label(frame, text = "Spot Price of Asset:").grid(row = 2, column = 1, sticky = W)
        Label(frame, text = "Risk-free Interest Rate:").grid(row = 3, column = 1, sticky = W)
        Label(frame, text = "Repo Rate:").grid(row = 4, column = 1, sticky = W)
        Label(frame, text = "Time to Maturity (in years):").grid(row = 5, column = 1, sticky = W)
        Label(frame, text = "Strike:").grid(row = 6, column = 1, sticky = W)
        Label(frame, text = "Option Premium:").grid(row = 7, column = 1, sticky = W)
        Label(frame, text = "Option Type:").grid(row = 8, column = 1, sticky = W)

        self.s0 = DoubleVar()
        self.s0.set(100)
        self.r = DoubleVar()
        self.r.set(0.01)
        self.q = DoubleVar()
        self.q.set(0)
        self.T = DoubleVar()
        self.T.set(0.5)
        self.K = DoubleVar()
        self.K.set(100)
        self.OP = DoubleVar()
        self.OP.set(10)
        self.option_type = StringVar()


        Entry(frame, textvariable = self.s0).grid(row = 2, column = 2, sticky = E)
        Entry(frame, textvariable = self.r).grid(row = 3, column = 2, sticky = E)
        Entry(frame, textvariable = self.q).grid(row = 4, column = 2, sticky = E)
        Entry(frame, textvariable = self.T).grid(row = 5, column = 2, sticky = E)
        Entry(frame, textvariable = self.K).grid(row = 6, column = 2, sticky = E)
        Entry(frame, textvariable = self.OP).grid(row = 7, column = 2, sticky = E)


        self.option_type_selector = ttk.Combobox(frame, width = 17, values = ("Select Option Type", "Call Option", "Put Option"), textvariable = self.option_type, postcommand = self.run_task2)
        self.option_type_selector.current(0)
        self.option_type_selector.grid(row = 8, column = 2, sticky = E)


        Button(frame, width = 23, text = "Reset", command = self.reset_task2).grid(row = 9, column = 1, columnspan = 1, sticky = E)


        Button(frame, width = 23, text = "Run", command = self.run_task2).grid(row = 9, column = 1, columnspan = 1, sticky = W)


        self.logs = scrolledtext.ScrolledText(frame, width = 74, height = 12)
        self.logs.grid(row = 10, column = 1, rowspan = 4, columnspan = 2, sticky = W)

    # closed-form formulas for geometric Asian call/put option.
    def task3(self):

        self.hide_all_frames()
        frame = self.frames['pricer3']
        frame.pack()

        Label(frame, text = " Closed-form Formulas for Geometric Asian Call/Put Options", justify = "right").grid(row = 1, column = 1,sticky = W)
        Label(frame, text = "Spot Price of Asset:").grid(row = 2, column = 1, sticky = W)
        Label(frame, text = "Implied Volatility:").grid(row = 3, column = 1, sticky = W)
        Label(frame, text = "Risk-free Interest Rate:").grid(row = 4, column = 1, sticky = W)
        Label(frame, text = "Time to Maturity (in years):").grid(row = 5, column = 1, sticky = W)
        Label(frame, text = "Strike:").grid(row = 6, column = 1, sticky = W)
        Label(frame, text = "Observation Times for the Geometric Average:").grid(row = 7, column = 1, sticky = W)
        Label(frame, text = "Option Type:").grid(row = 8, column = 1, sticky = W)

        self.s0 = DoubleVar()
        self.s0.set(100)
        self.sigma = DoubleVar()
        self.sigma.set(0.3)
        self.r = DoubleVar()
        self.r.set(0.05)
        self.T = DoubleVar()
        self.T.set(3)
        self.K = DoubleVar()
        self.K.set(100)
        self.n = IntVar()
        self.n.set(50)
        self.option_type = StringVar()


        Entry(frame, textvariable = self.s0).grid(row = 2, column = 2, sticky = E)
        Entry(frame, textvariable = self.sigma).grid(row = 3, column = 2, sticky = E)
        Entry(frame, textvariable = self.r).grid(row = 4, column = 2, sticky = E)
        Entry(frame, textvariable = self.T).grid(row = 5, column = 2, sticky = E)
        Entry(frame, textvariable = self.K).grid(row = 6, column = 2, sticky = E)
        Entry(frame, textvariable = self.n).grid(row = 7, column = 2, sticky = E)


        self.option_type_selector = ttk.Combobox(frame, width = 17, values = ("Select Option Type", "Call Option", "Put Option"), textvariable = self.option_type, postcommand = self.run_task3)
        self.option_type_selector.current(0)
        self.option_type_selector.grid(row = 8, column = 2, sticky = E)

        Button(frame, width = 29, text = "Reset", command = self.reset_task3).grid(row = 9, column = 1, columnspan = 1, sticky = E)

        Button(frame, width = 29, text = "Run", command = self.run_task3).grid(row = 9, column = 1, columnspan = 1, sticky = W)

        self.logs = scrolledtext.ScrolledText(frame, width = 74, height = 12)
        self.logs.grid(row = 10, column = 1, rowspan = 4, columnspan = 2, sticky = W)


    def task4(self):

        self.hide_all_frames()
        frame = self.frames['pricer4']
        frame.pack()

        Label(frame, text = " Closed-form Formulas for Geometric Basket Call/Put Options", justify = "right").grid(row = 1, column = 1,sticky = W)
        Label(frame, text = "Spot Price of Asset 1:").grid(row = 2, column = 1, sticky = W)
        Label(frame, text = "Spot Price of Asset 2:").grid(row = 3, column = 1, sticky = W)
        Label(frame, text = "Volatility of Asset 1:").grid(row = 4, column = 1, sticky = W)
        Label(frame, text = "Volatility of Asset 2:").grid(row = 5, column = 1, sticky = W)
        Label(frame, text = "Risk-free Interest Rate:").grid(row = 6, column = 1, sticky = W)
        Label(frame, text = "Time to Maturity (in year):").grid(row = 7, column = 1, sticky = W)
        Label(frame, text = "Strike:").grid(row = 8, column = 1, sticky = W)
        Label(frame, text = "Correlation:").grid(row = 9, column = 1, sticky = W)
        Label(frame, text = "Option Type:").grid(row = 10, column = 1, sticky = W)

        self.s0_1 = DoubleVar()
        self.s0_1.set(100)
        self.s0_2 = DoubleVar()
        self.s0_2.set(100)
        self.sigma_1 = DoubleVar()
        self.sigma_1.set(0.3)
        self.sigma_2 = DoubleVar()
        self.sigma_2.set(0.3)
        self.r = DoubleVar()
        self.r.set(0.05)
        self.T = DoubleVar()
        self.T.set(3)
        self.K = DoubleVar()
        self.K.set(100)
        self.rho = DoubleVar()
        self.rho.set(0.5)
        self.option_type = StringVar()


        Entry(frame, textvariable = self.s0_1).grid(row = 2, column = 2, sticky = E)
        Entry(frame, textvariable = self.s0_2).grid(row = 3, column = 2, sticky = E)
        Entry(frame, textvariable = self.sigma_1).grid(row = 4, column = 2, sticky = E)
        Entry(frame, textvariable = self.sigma_2).grid(row = 5, column = 2, sticky = E)
        Entry(frame, textvariable = self.r).grid(row = 6, column = 2, sticky = E)
        Entry(frame, textvariable = self.T).grid(row = 7, column = 2, sticky = E)
        Entry(frame, textvariable = self.K).grid(row = 8, column = 2, sticky = E)
        Entry(frame, textvariable = self.rho).grid(row = 9, column = 2, sticky = E)


        self.option_type_selector = ttk.Combobox(frame, width = 17, values = ("Select Option Type", "Call Option", "Put Option"), textvariable = self.option_type, postcommand = self.run_task4)
        self.option_type_selector.current(0)
        self.option_type_selector.grid(row = 10, column = 2, sticky = E)


        Button(frame, width = 29, text = "Reset", command = self.reset_task4).grid(row = 11, column = 1, columnspan = 1, sticky = E)


        Button(frame, width = 29, text = "Run", command = self.run_task4).grid(row = 11, column = 1, columnspan = 1, sticky = W)


        self.logs = scrolledtext.ScrolledText(frame, width = 74, height = 9)
        self.logs.grid(row = 12, column = 1, rowspan = 4, columnspan = 2, sticky = W)


    def task5(self):

        frame = self.frames['pricer5']
        self.hide_all_frames()
        frame.pack()

        Label(frame, text = "Arithmetic Asian Option from MC", justify = "right").grid(row = 1, column = 1,sticky = W)
        Label(frame, text="S0").grid(row=2, column=1, sticky=E)
        Label(frame, text="sigma").grid(row=2, column=3, sticky=E)
        Label(frame, text="r").grid(row=3, column=1, sticky=E)
        Label(frame, text="T").grid(row=3, column=3, sticky=E)
        Label(frame, text="n").grid(row=4, column=1, sticky=E)
        Label(frame, text="K").grid(row=4, column=3, sticky=E)
        Label(frame, text="m").grid(row=5, column=1, sticky=E)



        self.s0 = DoubleVar()
        self.s0.set(100)
        self.sigma = DoubleVar()
        self.sigma.set(0.3)
        self.r = DoubleVar()
        self.r.set(0.05)
        self.T = DoubleVar()
        self.T.set(3)
        self.K = DoubleVar()
        self.K.set(100)
        self.n = IntVar()
        self.n.set(50)

        self.m = IntVar()
        self.m.set(100000)
        self.ctrl = BooleanVar()
        self.option_type = StringVar()

        Entry(frame, width=15, textvariable=self.s0).grid(row=2, column=2, sticky=E)
        Entry(frame, textvariable=self.sigma).grid(row=2, column=4, sticky=E)
        Entry(frame, width=15,textvariable=self.r).grid(row=3, column=2, sticky=E)
        Entry(frame, textvariable=self.T).grid(row=3, column=4, sticky=E)
        Entry(frame, width=15, textvariable=self.n).grid(row=4, column=2, sticky=E)
        Entry(frame, textvariable=self.K).grid(row=4, column=4, sticky=E)
        Entry(frame, width=15, textvariable=self.m).grid(row=5, column=2, sticky=E)

        Checkbutton(frame, text="Control Variate:", variable=self.ctrl).grid(row=6, column=1, sticky=W)

        Radiobutton(frame, width=6, text="Put", variable=self.option_type, value='put').grid(row=6, column=2, sticky=E)
        Radiobutton(frame, width=6, text="Call", variable=self.option_type, value='call').grid(row=6, column=3, sticky=W)

        Button(frame, width=10, text="Run", command=self.run_task5).grid(row=7, column=1, columnspan=4)
        Button(frame, width=10, text="Reset", command=self.reset_task5).grid(row=7, column=1, columnspan=4, sticky=E)

        self.logs = scrolledtext.ScrolledText(frame, width = 74, height = 16)
        self.logs.grid(row=8, column=1, columnspan=4)


    def task6(self):

        frame = self.frames['pricer6']
        self.hide_all_frames()
        frame.pack()

        Label(frame, text = "Arithmetic Mean Bakset Option from MC", justify = "right").grid(row = 1, column = 1,sticky = W)
        Label(frame, text="S0_1").grid(row=2, column=1, sticky=E)
        Label(frame, text="S0_2").grid(row=2, column=3, sticky=E)
        Label(frame, text="sigma_1").grid(row=3, column=1, sticky=E)
        Label(frame, text="sigma_2").grid(row=3, column=3, sticky=E)
        Label(frame, text="r").grid(row=4, column=1, sticky=E)
        Label(frame, text="T").grid(row=4, column=3, sticky=E)
        Label(frame, text="K").grid(row=5, column=1, sticky=E)
        Label(frame, text="rho").grid(row=5, column=3, sticky=E)
        Label(frame, text="m").grid(row=6, column=1, sticky=E)


        self.s0_1 = DoubleVar()
        self.s0_1.set(100)
        self.s0_2 = DoubleVar()
        self.s0_2.set(100)
        self.sigma_1 = DoubleVar()
        self.sigma_1.set(0.3)
        self.sigma_2 = DoubleVar()
        self.sigma_2.set(0.3)
        self.r = DoubleVar()
        self.r.set(0.05)
        self.T = DoubleVar()
        self.T.set(3)
        self.K = DoubleVar()
        self.K.set(100)
        self.rho = DoubleVar()
        self.rho.set(0.5)
        self.m = IntVar()
        self.m.set(100000)
        self.ctrl = BooleanVar()
        self.option_type = StringVar()

        Entry(frame, width=16, textvariable=self.s0_1).grid(row=2, column=2, sticky=E)
        Entry(frame, width=20, textvariable=self.s0_2).grid(row=2, column=4, sticky=E)
        Entry(frame, width=16, textvariable=self.sigma_1).grid(row=3, column=2, sticky=E)
        Entry(frame, width=20, textvariable=self.sigma_2).grid(row=3, column=4, sticky=E)
        Entry(frame, width=16, textvariable=self.r).grid(row=4, column=2, sticky=E)
        Entry(frame, width=20, textvariable=self.T).grid(row=4, column=4, sticky=E)
        Entry(frame, width=16, textvariable=self.K).grid(row=5, column=2, sticky=E)
        Entry(frame, width=20, textvariable=self.rho).grid(row=5, column=4, sticky=E)
        Entry(frame, width=16, textvariable=self.m).grid(row=6, column=2, sticky=E)

        Checkbutton(frame, text="Control Variate:", variable=self.ctrl).grid(row=7, column=1)

        Radiobutton(frame, text="Put",  variable=self.option_type, value='put').grid(row=7, column=2)
        Radiobutton(frame, text="Call", variable=self.option_type, value='call').grid(row=7, column=3)

        Button(frame, width=10, text="Run", command=self.run_task6).grid(row=8, column=1, columnspan=2, sticky=E)
        Button(frame, width=10, text="Reset", command=self.reset_task6).grid(row=8, column=1, columnspan=4, sticky=E)

        self.logs = scrolledtext.ScrolledText(frame, height=14)
        self.logs.grid(row=9, column=1, columnspan=4)


    def task7(self):

        frame = self.frames['pricer7']
        self.hide_all_frames()
        frame.pack()


        Label(frame, text = "Binomial Tree Method for American Call/Put Options", fg = "red", justify = "right").grid(row = 1, column = 1,sticky = W)
        Label(frame, text = "Spot Price of Asset:").grid(row = 2, column = 1, sticky = W)
        Label(frame, text = "Volatility:").grid(row = 3, column = 1, sticky = W)
        Label(frame, text = "Risk-free Interest Rate:").grid(row = 4, column = 1, sticky = W)
        Label(frame, text = "Time to Maturity (in years):").grid(row = 5, column = 1, sticky = W)
        Label(frame, text = "Strike:").grid(row = 6, column = 1, sticky = W)
        Label(frame, text = "Number of Steps:").grid(row = 7, column = 1, sticky = W)
        Label(frame, text = "Option Type:").grid(row = 8, column = 1, sticky = W)

        self.s0 = DoubleVar()
        self.s0.set(100)
        self.sigma = DoubleVar()
        self.sigma.set(0.3)
        self.r = DoubleVar()
        self.r.set(0.05)
        self.T = DoubleVar()
        self.T.set(3)
        self.K = DoubleVar()
        self.K.set(100)
        self.N = IntVar()
        self.N.set(36)
        self.option_type = StringVar()


        Entry(frame, textvariable = self.s0).grid(row = 2, column = 2, sticky = W)
        Entry(frame, textvariable = self.sigma).grid(row = 3, column = 2, sticky = W)
        Entry(frame, textvariable = self.r).grid(row = 4, column = 2, sticky = W)
        Entry(frame, textvariable = self.T).grid(row = 5, column = 2, sticky = W)
        Entry(frame, textvariable = self.K).grid(row = 6, column = 2, sticky = W)
        Entry(frame, textvariable = self.N).grid(row = 7, column = 2, sticky = W)


        self.option_type_selector = ttk.Combobox(frame, width = 17, values = ("Select Option Type", "Call Option", "Put Option"), textvariable = self.option_type, postcommand = self.run_task7)
        self.option_type_selector.current(0)
        self.option_type_selector.grid(row = 8, column = 2, sticky = W)


        Button(frame, width = 10, text = "Reset", command = self.reset_task7).grid(row = 9, column = 2, columnspan = 1, sticky = E)


        Button(frame, width = 10, text = "Run", command = self.run_task7).grid(row = 9, column = 2, columnspan = 1, sticky = W)


        self.logs = scrolledtext.ScrolledText(frame, width = 74, height = 12)
        self.logs.grid(row = 10, column = 1, rowspan = 4, columnspan = 2, sticky = W)

    #  the Quasi Monte Carlo method for for a KIKO-put option.
    def task8(self):

        frame = self.frames['pricer8']
        self.hide_all_frames()
        frame.pack()

        Label(frame, text = "KIKO-put option from MC", fg = "red", justify = "right").grid(row = 1, column = 1,sticky = W)
        Label(frame, text="Spot Price of Asset:").grid(row=2, column=1, sticky=W)
        Label(frame, text="Volatility:").grid(row=3, column=1, sticky=W)
        Label(frame, text="Risk-free Interest Rate:").grid(row=4, column=1, sticky=W)
        Label(frame, text="Time to Maturity (in years):").grid(row=5, column=1, sticky=W)
        Label(frame, text="Strike:").grid(row=6, column=1, sticky=W)
        Label(frame, text="Barrier Lower").grid(row=7, column=1, sticky=W)
        Label(frame, text="Barrier Upper").grid(row=8, column=1, sticky=W)
        Label(frame, text="Number of Observations").grid(row=9, column=1, sticky=W)
        Label(frame, text="Cash Rebate").grid(row=10, column=1, sticky=W)

        self.S = DoubleVar()
        self.S.set(100)
        self.sigma = DoubleVar()
        self.sigma.set(0.2)
        self.r = DoubleVar()
        self.r.set(0.05)
        self.T = DoubleVar()
        self.T.set(2)
        self.K = DoubleVar()
        self.K.set(100)
        self.barrier_lower = DoubleVar()
        self.barrier_lower.set(80)
        self.barrier_upper = DoubleVar()
        self.barrier_upper.set(125)
        self.N = IntVar()
        self.N.set(24)
        self.R = DoubleVar()
        self.R.set(1.5)


        Entry(frame, textvariable = self.S).grid(row = 2, column = 2, sticky = E)
        Entry(frame, textvariable = self.sigma).grid(row = 3, column = 2, sticky = E)
        Entry(frame, textvariable = self.r).grid(row = 4, column = 2, sticky = E)
        Entry(frame, textvariable = self.T).grid(row = 5, column = 2, sticky = E)
        Entry(frame, textvariable = self.K).grid(row = 6, column = 2, sticky = E)
        Entry(frame, textvariable = self.barrier_lower).grid(row = 7, column = 2, sticky = E)
        Entry(frame, textvariable = self.barrier_upper).grid(row = 8, column = 2, sticky = E)
        Entry(frame, textvariable = self.N).grid(row = 9, column = 2, sticky = E)
        Entry(frame, textvariable = self.R).grid(row = 10, column = 2, sticky = E)

        Button(frame, width=10, text="Reset", command=self.reset_task7).grid(row=12, column=1, columnspan=2, sticky=E)
        Button(frame, width=10, text="Run", command=self.run_task8).grid(row=12, column=1, columnspan=1, sticky=E)

        self.logs = scrolledtext.ScrolledText(frame, height=14)
        self.logs.grid(row=13, column=1, columnspan=4)


    def run_homepage(self):
        
        self.hide_all_frames()
        self.frames['home'].pack()
        
    def run_task1(self):
        
        OptionType = self.option_type.get()
        
        if OptionType == "Call Option":
            
            try:
  
      
                result = black_scholes(S = self.s0.get(), sigma = self.sigma.get(), r = self.r.get(), q = self.repo.get(), T = self.T.get(), K = self.K.get(), call=True)
                self.logs.insert(END, "The Call Option Premium is: {}\n".format(result))
                
            except ZeroDivisionError:
                
                self.logs.insert(END, "Input Parameter Error! Please input the correct parameters!\n")
                
        elif OptionType == "Put Option": 
            
            try:
            
                
                result = black_scholes(S = self.s0.get(), sigma = self.sigma.get(), r = self.r.get(), q = self.repo.get(), T = self.T.get(), K = self.K.get(), call=False)
                self.logs.insert(END, "The Put Option Premium is: {}\n".format(result))
            
            except ZeroDivisionError:
                
                self.logs.insert(END, "Input Parameter Error! Please input the correct parameters!\n")
        
        else:
            
            pass
        
        self.option_type_selector.current(0)
            
    def reset_task1(self):
        self.task1()
        
    def run_task2(self):
        
        OptionType = self.option_type.get()
        
        if OptionType == "Call Option":
            
            try:
  
                result = Newton_Raphson4Ivolatility(S = self.s0.get(), r = self.r.get(), q = self.q.get(), T = self.T.get(), K = self.K.get(), C_true = self.OP.get(), call=True)
               
                
                if math.isnan(result) or math.isinf(result):
                    
                    self.logs.insert(END, "Input Parameter Error! Please input the correct parameters!\n")
                    
                else:
                    
                    self.logs.insert(END, "The Implied Volatility for Call Option is: {}\n".format(result))
                
            except ZeroDivisionError:
                
                self.logs.insert(END, "Input Parameter Error! Please input the correct parameters!\n")
                
        if OptionType == "Put Option": 
            
            try:
            
                result = Newton_Raphson4Ivolatility(S = self.s0.get(), r = self.r.get(), q = self.q.get(), T = self.T.get(), K = self.K.get(), C_true = self.OP.get(), call=False)
                
                if math.isnan(result) or math.isinf(result):
                    
                    self.logs.insert(END, "Input Parameter Error! Please input the correct parameters!\n")
                    
                else:
                    
                    self.logs.insert(END, "The Implied Volatility for Put Option is: {}\n".format(result))
            
            except ZeroDivisionError:
                
                self.logs.insert(END, "Input Parameter Error! Please input the correct parameters!\n")
                
        self.option_type_selector.current(0)
            
    def reset_task2(self):
        
        self.task2()
        
    def run_task3(self):
        
        OptionType = self.option_type.get()
        
        if OptionType == "Call Option":
            
            try:
  
                result = geometric_asia_option(S0 = self.s0.get(), sigma = self.sigma.get(), r = self.r.get(), T = self.T.get(), K = self.K.get(), n = self.n.get(), option_type = 'call')
                
                
                if math.isnan(result) or math.isinf(result):
                    
                    self.logs.insert(END, "Input Parameter Error! Please input the correct parameters!\n")
                    
                else:
                    
                    self.logs.insert(END, "The Call Option Premium is: {}\n".format(result))
                
            except ZeroDivisionError:
                
                self.logs.insert(END, "Input Parameter Error! Please input the correct parameters!\n")
                
        if OptionType == "Put Option": 
            
            try:
            
                result = geometric_asia_option(S0 = self.s0.get(), sigma = self.sigma.get(), r = self.r.get(), T = self.T.get(), K = self.K.get(), n = self.n.get(), option_type = 'put')
                
                if math.isnan(result) or math.isinf(result):
                    
                    self.logs.insert(END, "Input Parameter Error! Please input the correct parameters!\n")
                    
                else:
                    
                    self.logs.insert(END, "The Put Option Premium is: {}\n".format(result))
            
            except ZeroDivisionError:
                
                self.logs.insert(END, "Input Parameter Error! Please input the correct parameters!\n")
                
        self.option_type_selector.current(0)
        
    def reset_task3(self):
        
        self.task3()
        
    def run_task4(self):
        
        OptionType = self.option_type.get()
        
        if OptionType == "Call Option":
            
            try:
  
                result = geometric_bascket_option(S1 = self.s0_1.get(), S2 = self.s0_2.get(), 
                                          sigma1 = self.sigma_1.get(), sigma2 = self.sigma_2.get(), 
                                          r = self.r.get(), T = self.T.get(), K = self.K.get(), rho = self.rho.get(), option_type = 'call')
          
                
                if math.isnan(result) or math.isinf(result):
                    
                    self.logs.insert(END, "Input Parameter Error! Please input the correct parameters!\n")
                    
                else:
                    
                    self.logs.insert(END, "The Call Option Premium is: {}\n".format(result))
                
            except ZeroDivisionError:
                
                self.logs.insert(END, "Input Parameter Error! Please input the correct parameters!\n")
                
        if OptionType == "Put Option": 
            
            try:
            
                result = geometric_bascket_option(S1 = self.s0_1.get(), S2 = self.s0_2.get(), 
                                          sigma1 = self.sigma_1.get(), sigma2 = self.sigma_2.get(), 
                                          r = self.r.get(), T = self.T.get(), K = self.K.get(), rho = self.rho.get(), option_type = 'put')
                
                if math.isnan(result) or math.isinf(result):
                    
                    self.logs.insert(END, "Input Parameter Error! Please input the correct parameters!\n")
                    
                else:
                    
                    self.logs.insert(END, "The Put Option Premium is: {}\n".format(result))
            
            except ZeroDivisionError:
                
                self.logs.insert(END, "Input Parameter Error! Please input the correct parameters!\n")
                
        self.option_type_selector.current(0)
    
    def reset_task4(self):
        
        self.task4()
        
    def run_task5(self):
        if self.ctrl.get():
            cv = 'control variate'
        else:
            cv = ''
        result = arithmetic_asia_option(S0=self.s0.get(), sigma=self.sigma.get(), r=self.r.get(),
                                    T=self.T.get(), K=self.K.get(), n=self.n.get(), M=self.m.get(), 
                                    option_type=self.option_type.get(), cv=cv)
        self.logs.insert(END, "The option premium is: {}\n".format(result))

    def reset_task5(self):

        self.task5()

    def run_task6(self):
        if self.ctrl.get():
            cv = 'control variate'
        else:
            cv = ''

        result = arithmetic_bascket_option(S1=self.s0_1.get() ,S2=self.s0_2.get(), sigma1=self.sigma_1.get(),
                                     sigma2=self.sigma_2.get(), r=self.r.get(), T=self.T.get(), K=self.K.get(),
                                     rho=self.rho.get(),option_type=self.option_type.get(),M=self.m.get(),
                                     cv=cv)
        
        self.logs.insert(END, "The put option premium is: {}\n".format(result))

    def reset_task6(self):

        self.task6()

    def run_task7(self):
        
        OptionType = self.option_type.get()
        
        if OptionType == "Call Option":
            
            try:
  
                
                result = american_option(S0 = self.s0.get(), sigma = self.sigma.get(), r = self.r.get(), T = self.T.get(), K = self.K.get(), N = self.N.get(), option_type = 'call')
                self.logs.insert(END, "The Call Option Premium is: {}\n".format(result))
                
            except ZeroDivisionError:
                
                self.logs.insert(END, "Input Parameter Error! Please input the correct parameters!\n")
                
        elif OptionType == "Put Option": 
            
            try:
            
                result = american_option(S0 = self.s0.get(), sigma = self.sigma.get(), r = self.r.get(), T = self.T.get(), K = self.K.get(), N = self.N.get(), option_type = 'put')
                self.logs.insert(END, "The Put Option Premium is: {}\n".format(result))
            
            except ZeroDivisionError:
                
                self.logs.insert(END, "Input Parameter Error! Please input the correct parameters!\n")
        
        else:
            
            pass
        
        self.option_type_selector.current(0)
    
    def reset_task7(self):
        
        self.task7()

    def run_task8(self):
        
        self.logs.insert(END, "waiting.... [It may take you several minutes]\n\n")

        result = Quasi_Monte_Carlo(S = self.S.get(), K= self.K.get(), T= self.T.get(), r= self.r.get(), sigma= self.sigma.get(), N= self.N.get(),
                                   R= self.R.get(), barrier_lower= self.barrier_lower.get(), barrier_upper= self.barrier_upper.get(), is_reversed = False)
        
        self.logs.insert(END, "The put option premium is: {}, the upper bound is {} and the lower bound is {}\n".format(result[0], result[1], result[2]))

    def reset_task8(self):

        self.task8()

    def Quit(self):
        
        self.window.destroy()

if __name__ == '__main__':
    
    OptionPricerGUI()