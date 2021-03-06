# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'design.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from networkx.algorithms.shortest_paths import weighted
import GraphReader

class Ui_Algorithms(object):
    def setupUi(self, Algorithms):
        Algorithms.setObjectName("Algorithms")
        Algorithms.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(Algorithms)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, -30, 801, 601))
        self.tabWidget.setObjectName("tabWidget")
        self.BrowseFile = QtWidgets.QWidget()
        self.BrowseFile.setObjectName("BrowseFile")
        self.label = QtWidgets.QLabel(self.BrowseFile)
        self.label.setGeometry(QtCore.QRect(120, 60, 551, 61))
        self.label.setStyleSheet("font: italic 36pt \"Monotype Corsiva\";")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.BrowseFile)
        self.label_2.setGeometry(QtCore.QRect(190, 120, 381, 41))
        self.label_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_2.setStyleSheet("font: 12pt \"MS Shell Dlg 2\";")
        self.label_2.setObjectName("label_2")
        self.filename = QtWidgets.QLineEdit(self.BrowseFile)
        self.filename.setGeometry(QtCore.QRect(110, 280, 361, 21))
        self.filename.setObjectName("filename")
        self.browseFile = QtWidgets.QPushButton(self.BrowseFile)
        self.browseFile.setGeometry(QtCore.QRect(510, 280, 141, 23))
        self.browseFile.setObjectName("browseFile")
        self.label_3 = QtWidgets.QLabel(self.BrowseFile)
        self.label_3.setGeometry(QtCore.QRect(110, 250, 231, 31))
        self.label_3.setStyleSheet("font: 11pt \"MS Shell Dlg 2\";")
        self.label_3.setObjectName("label_3")
        self.MovetoAlgo = QtWidgets.QPushButton(self.BrowseFile)
        self.MovetoAlgo.setGeometry(QtCore.QRect(330, 390, 101, 31))
        self.MovetoAlgo.setObjectName("MovetoAlgo")
        self.tabWidget.addTab(self.BrowseFile, "")
        self.ChooseAlgo = QtWidgets.QWidget()
        self.ChooseAlgo.setObjectName("ChooseAlgo")
        self.label_4 = QtWidgets.QLabel(self.ChooseAlgo)
        self.label_4.setGeometry(QtCore.QRect(280, 260, 231, 31))
        self.label_4.setStyleSheet("font: 12pt \"MS Shell Dlg 2\";")
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.ChooseAlgo)
        self.label_5.setGeometry(QtCore.QRect(120, 60, 551, 61))
        self.label_5.setStyleSheet("font: italic 36pt \"Monotype Corsiva\";")
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.ChooseAlgo)
        self.label_6.setGeometry(QtCore.QRect(190, 120, 381, 41))
        self.label_6.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_6.setStyleSheet("font: 12pt \"MS Shell Dlg 2\";")
        self.label_6.setObjectName("label_6")
        self.Prims = QtWidgets.QPushButton(self.ChooseAlgo)
        self.Prims.setGeometry(QtCore.QRect(170, 300, 161, 61))
        self.Prims.setObjectName("Prims")
        self.Kruskal = QtWidgets.QPushButton(self.ChooseAlgo)
        self.Kruskal.setGeometry(QtCore.QRect(460, 300, 161, 61))
        self.Kruskal.setObjectName("Kruskal")
        self.Dijkstra = QtWidgets.QPushButton(self.ChooseAlgo)
        self.Dijkstra.setGeometry(QtCore.QRect(170, 370, 161, 61))
        self.Dijkstra.setObjectName("Dijkstra")
        self.Warshall = QtWidgets.QPushButton(self.ChooseAlgo)
        self.Warshall.setGeometry(QtCore.QRect(170, 440, 161, 61))
        self.Warshall.setObjectName("Warshall")
        self.Boruvka = QtWidgets.QPushButton(self.ChooseAlgo)
        self.Boruvka.setGeometry(QtCore.QRect(310, 510, 161, 61))
        self.Boruvka.setObjectName("Boruvka")
        self.Cluster = QtWidgets.QPushButton(self.ChooseAlgo)
        self.Cluster.setGeometry(QtCore.QRect(460, 440, 161, 61))
        self.Cluster.setObjectName("Cluster")
        self.Bellman = QtWidgets.QPushButton(self.ChooseAlgo)
        self.Bellman.setGeometry(QtCore.QRect(460, 370, 161, 61))
        self.Bellman.setObjectName("Bellman")
        self.DirNode = QtWidgets.QLabel(self.ChooseAlgo)
        self.DirNode.setGeometry(QtCore.QRect(40, 180, 381, 21))
        self.DirNode.setStyleSheet("font: 12pt \"MS Shell Dlg 2\";")
        self.DirNode.setObjectName("DirNode")
        self.DirNode_2 = QtWidgets.QLabel(self.ChooseAlgo)
        self.DirNode_2.setGeometry(QtCore.QRect(40, 210, 381, 21))
        self.DirNode_2.setStyleSheet("font: 12pt \"MS Shell Dlg 2\";")
        self.DirNode_2.setObjectName("DirNode_2")
        self.tabWidget.addTab(self.ChooseAlgo, "")
        self.Result = QtWidgets.QWidget()
        self.Result.setObjectName("Result")
        self.Resultimage = QtWidgets.QLabel(self.Result)
        self.Resultimage.setGeometry(QtCore.QRect(300, 260, 401, 261))
        self.Resultimage.setObjectName("Resultimage")
        self.label_7 = QtWidgets.QLabel(self.Result)
        self.label_7.setGeometry(QtCore.QRect(80, 400, 111, 31))
        self.label_7.setStyleSheet("font: 12pt \"MS Shell Dlg 2\";\n"
"")
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.Result)
        self.label_8.setGeometry(QtCore.QRect(430, 90, 111, 31))
        self.label_8.setStyleSheet("font: 12pt \"MS Shell Dlg 2\";\n"
"")
        self.label_8.setObjectName("label_8")
        self.Resultimage_2 = QtWidgets.QLabel(self.Result)
        self.Resultimage_2.setGeometry(QtCore.QRect(20, 20, 381, 211))
        self.Resultimage_2.setObjectName("Resultimage_2")
        self.tabWidget.addTab(self.Result, "")
        Algorithms.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Algorithms)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        Algorithms.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Algorithms)
        self.statusbar.setObjectName("statusbar")
        Algorithms.setStatusBar(self.statusbar)

        self.retranslateUi(Algorithms)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Algorithms)

    def retranslateUi(self, Algorithms):
        _translate = QtCore.QCoreApplication.translate
        Algorithms.setWindowTitle(_translate("Algorithms", "Algorithms & Designs"))
        self.label.setText(_translate("Algorithms", "Design Analysis & Algorithms"))
        self.label_2.setText(_translate("Algorithms", "Run any algorithm on any number of nodes and files"))
        self.browseFile.setText(_translate("Algorithms", "Browse"))
        self.label_3.setText(_translate("Algorithms", "Browse or enter the name of file :"))
        self.MovetoAlgo.setText(_translate("Algorithms", "Next"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.BrowseFile), _translate("Algorithms", "Browse File"))
        self.label_4.setText(_translate("Algorithms", "Press the Algorithms to generate the results"))
        self.label_5.setText(_translate("Algorithms", "Design Analysis & Algorithms"))
        self.label_6.setText(_translate("Algorithms", "Run any algorithm on any number of nodes and files"))
        self.Prims.setText(_translate("Algorithms", "Prim\'s Algorithm"))
        self.Kruskal.setText(_translate("Algorithms", "Kruskal\'s Algorithm"))
        self.Dijkstra.setText(_translate("Algorithms", "Dijkstra\'s Algorithm"))
        self.Warshall.setText(_translate("Algorithms", "Floyd Warshall"))
        self.Boruvka.setText(_translate("Algorithms", "Boruvka\'s Algorithm"))
        self.Cluster.setText(_translate("Algorithms", "Clustering Coefficients"))
        self.Bellman.setText(_translate("Algorithms", "Bellman Ford"))
        self.DirNode.setText(_translate("Algorithms", "directed graph"))
        self.DirNode_2.setText(_translate("Algorithms", "Undirected graph"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.ChooseAlgo), _translate("Algorithms", "Choose Algorithms"))
        self.Resultimage.setText(_translate("Algorithms", "TextLabel"))
        self.label_7.setText(_translate("Algorithms", "Resulted Graph"))
        self.label_8.setText(_translate("Algorithms", "Graph from file"))
        self.Resultimage_2.setText(_translate("Algorithms", "TextLabel"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Result), _translate("Algorithms", "Result"))

    def SelectFile(self):
        Location = "E:\Algorithms\Algorithms-Design"
        global fname
        fname = QFileDialog.getOpenFileName(None, "Select File", Location, "Text files (*.txt)")
        self.filename.setText(fname[0])

    def MovetoAlgoPage(self):
        self.tabWidget.setCurrentIndex(1)
        GraphReader.readInputFile(fname[0])
        self.DirNode.setText("Total Nodes in Directed Graph : " + str(GraphReader.di_verts))
        self.DirNode_2.setText("Total Nodes in UnDirected Graph : " + str(GraphReader.di_verts))

    def doPrims(self):
        weight = GraphReader.PrimAlgo()
        self.tabWidget.setCurrentIndex(2)
        self.Resultimage.setPixmap(QtGui.QPixmap("PrimMST.png"))
        self.Resultimage.setScaledContents(True)
        self.Resultimage_2.setPixmap(QtGui.QPixmap("Graph.png"))
        self.Resultimage_2.setScaledContents(True)

        

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Algorithms = QtWidgets.QMainWindow()
    ui = Ui_Algorithms()
    ui.setupUi(Algorithms)
    Algorithms.show()

    ui.tabWidget.setCurrentIndex(0)

    ui.browseFile.clicked.connect(ui.SelectFile)
    ui.MovetoAlgo.clicked.connect(ui.MovetoAlgoPage)
    ui.Prims.clicked.connect(ui.doPrims)
    # ui.browseFile.clicked.connect()

    sys.exit(app.exec_())