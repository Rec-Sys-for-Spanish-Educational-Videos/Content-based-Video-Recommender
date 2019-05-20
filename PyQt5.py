# -*- coding: utf-8 -*-
"""
Created on Sat May 18 16:09:43 2019

@author: micke
"""

from PyQt5 import *
from gui import Ui_MainWindow
import sys
from Source import *

'''
UIClass, QtBaseClass = uic.loadUiType("GUI.ui")

class MyApp(UIClass, QtBaseClass):
    def __init__(self):
        UIClass.__init__(self)
        QtBaseClass.__init__(self)
        self.setupUi(self)

app = QtWidgets.QApplication(sys.argv)
window = MyApp()
window.show()
sys.exit(app.exec_())

'''

def on_button_clicked():
    query = application.ui.textEdit.toPlainText()
    results = resultForQuery(query)
    application.ui.textBrowser.setText('Ranked list of transcripts (transcript ID and score)\n'+results)

class mywindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(mywindow, self).__init__()    
        self.ui = Ui_MainWindow()    
        self.ui.setupUi(self)    
        #self.ui.label.setFont(QtGui.QFont('SansSerif', 30)) # change font type and size
        
app = QtWidgets.QApplication([])
application = mywindow()
application.ui.pushButton.clicked.connect(on_button_clicked)
application.show()
sys.exit(app.exec())