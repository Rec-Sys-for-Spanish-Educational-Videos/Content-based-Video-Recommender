# -*- coding: utf-8 -*-
"""
Created on Sat May 18 16:09:43 2019

@author: micke
"""

from PyQt5 import *
from gui import Ui_MainWindow
import sys
from Source import *

# Function that gets called when the "Search" button is pressed
def on_button_clicked():
    query = application.ui.textEdit.toPlainText()
    results = resultForQuery(query)
    application.ui.textBrowser.setText('Ranked list of transcripts (transcript ID and score)\n'+results)

#The class for the QT5 Window
class mywindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(mywindow, self).__init__()    
        self.ui = Ui_MainWindow()    
        self.ui.setupUi(self)    

#Code that initializes the GUI  
app = QtWidgets.QApplication([])
application = mywindow()
application.ui.pushButton.clicked.connect(on_button_clicked)
application.show()
sys.exit(app.exec())