from os import read
import tkinter as tk
from tkinter import ttk
from tkinter import *
import GraphReader


def openGraphs():
    pass

def on_closing():
    root.destroy()

def openNewWindow():
    root.withdraw() 
    # Toplevel object which will
    # be treated as a new window
    global newWindow
    newWindow = Toplevel(root)
    newWindow.protocol("WM_DELETE_WINDOW", on_closing)
 
    # sets the title of the
    # Toplevel widget
    newWindow.title("Algorithms")
 
    # sets the geometry of toplevel
    newWindow.geometry("450x350")
    newWindow.configure(background='#F0F8FF')
 
    # A Label widget to show in toplevel
    Label(newWindow, text='Select Algorithm:', bg='#F0F8FF',
      font=('arial', 24, 'normal')).place(x=113, y=38)

    Button(newWindow, text='Back', bg='#FFA07A', font=('arial', 12, 'normal'),
       command=SecondWindowBackClickFunction).place(x=10, y=10)

    # This is the section of code which creates a button
    Button(newWindow, text='Prim\'s', bg='#FFA07A', font=('arial', 12, 'normal'),
        command=GraphReader.PrimAlgo).place(x=83-5, y=118)

    # This is the section of code which creates a button
    Button(newWindow, text='Kruskal', bg='#FFA07A', font=('arial', 12, 'normal'),
        command=GraphReader.KruskalAlgo).place(x=193-5, y=118)


    # This is the section of code which creates a button
    Button(newWindow, text='Dijkstra', bg='#FFA07A', font=('arial', 12, 'normal'),
        command=GraphReader.DijkstraAlgo).place(x=303-5, y=118)


    # This is the section of code which creates a button
    Button(newWindow, text='Bellman Ford', bg='#FFA07A', font=('arial', 12, 'normal'),
        command=GraphReader.BellmanFordAlgo).place(x=83-5, y=178)


    # This is the section of code which creates a button
    Button(newWindow, text='Floyd Warshall', bg='#FFA07A', font=('arial', 12, 'normal'),
        command=GraphReader.FloydWarshallAlgo).place(x=255-5, y=178)


    # This is the section of code which creates a button
    Button(newWindow, text='Clustering Coefficient', bg='#FFA07A', font=('arial', 12, 'normal'),
        command=GraphReader.ClusteringCoefficientAlgo).place(x=83-5, y=238)


    # This is the section of code which creates a button
    Button(newWindow, text='Boruvka', bg='#FFA07A', font=('arial', 12, 'normal'),
        command=GraphReader.BoruvkaAlgo).place(x=303-5, y=238)


# All Algorithm Functions
# def PrimAlgo():
#     primG = GraphReader.PrimGraph(GraphReader.verts, GraphReader.starting)
#     primG.graph = GraphReader.graphMat
#     primG.primMST()
#     openGraphs()

def DijkstraAlgo():
    pass

def BellmanFordAlgo():
    pass

def FloydWarshallAlgo():
    pass

def BoruvkaAlgo():
    pass

# All button functions

def SecondWindowBackClickFunction():
    root.deiconify()
    newWindow.destroy()

def TenNodeClickFunction():
    input_file = "input10.txt"
    GraphReader.readInputFile(input_file) 
    openNewWindow()

# this is the function called when the button is clicked
def TwentyNodeClickFunction():
    input_file = "input20.txt"
    GraphReader.readInputFile(input_file) 
    openNewWindow()


# this is the function called when the button is clicked
def ThirtyNodeClickFunction():
    input_file = "input30.txt"
    GraphReader.readInputFile(input_file) 
    openNewWindow()


# this is the function called when the button is clicked
def FortyNodeClickFunction():
    input_file = "input40.txt"
    GraphReader.readInputFile(input_file) 
    openNewWindow()


# this is the function called when the button is clicked
def FiftyNodeClickFunction():
    input_file = "input50.txt"
    GraphReader.readInputFile(input_file) 
    openNewWindow()


# this is the function called when the button is clicked
def SixtyNodeClickFunction():
    input_file = "input60.txt"
    GraphReader.readInputFile(input_file) 
    openNewWindow()


# this is the function called when the button is clicked
def SeventyNodeClickFunction():
    input_file = "input70.txt"
    GraphReader.readInputFile(input_file) 
    openNewWindow()


# this is the function called when the button is clicked
def EightyNodeClickFunction():
    input_file = "input80.txt"
    GraphReader.readInputFile(input_file) 
    openNewWindow()


# this is the function called when the button is clicked
def NinetyNodeClickFunction():
    input_file = "input90.txt"
    GraphReader.readInputFile(input_file) 
    openNewWindow()


# this is the function called when the button is clicked
def HundredNodeClickFunction():
    input_file = "input100.txt"
    GraphReader.readInputFile(input_file) 
    openNewWindow()


input_file = str()
root = Tk()

# This is the section of code which creates the main window
root.geometry('450x350')
root.configure(background='#F0F8FF')
root.title('Graph Algorithms by 19K0181 and 19K1512')


# This is the section of code which creates the a label
Label(root, text='Select Input File:', bg='#F0F8FF',
      font=('arial', 24, 'normal')).place(x=113, y=38)


# This is the section of code which creates a button
Button(root, text='10 Nodes', bg='#FFA07A', font=('arial', 12, 'normal'),
       command=TenNodeClickFunction).place(x=83-5, y=118)


# This is the section of code which creates a button
Button(root, text='20 Nodes', bg='#FFA07A', font=('arial', 12, 'normal'),
       command=TwentyNodeClickFunction).place(x=193-5, y=118)


# This is the section of code which creates a button
Button(root, text='30 Nodes', bg='#FFA07A', font=('arial', 12, 'normal'),
       command=ThirtyNodeClickFunction).place(x=303-5, y=118)


# This is the section of code which creates a button
Button(root, text='40 Nodes', bg='#FFA07A', font=('arial', 12, 'normal'),
       command=FortyNodeClickFunction).place(x=83-5, y=178)


# This is the section of code which creates a button
Button(root, text='50 Nodes', bg='#FFA07A', font=('arial', 12, 'normal'),
       command=FiftyNodeClickFunction).place(x=193-5, y=178)


# This is the section of code which creates a button
Button(root, text='60 Nodes', bg='#FFA07A', font=('arial', 12, 'normal'),
       command=SixtyNodeClickFunction).place(x=303-5, y=178)


# This is the section of code which creates a button
Button(root, text='70 Nodes', bg='#FFA07A', font=('arial', 12, 'normal'),
       command=SeventyNodeClickFunction).place(x=83-5, y=238)


# This is the section of code which creates a button
Button(root, text='80 Nodes', bg='#FFA07A', font=('arial', 12, 'normal'),
       command=EightyNodeClickFunction).place(x=193-5, y=238)


# This is the section of code which creates a button
Button(root, text='90 Nodes', bg='#FFA07A', font=('arial', 12, 'normal'),
       command=NinetyNodeClickFunction).place(x=303-5, y=238)


# This is the section of code which creates a button
Button(root, text='100 Nodes', bg='#FFA07A', font=('arial', 12, 'normal'),
       command=HundredNodeClickFunction).place(x=193-9, y=298)


root.mainloop()
