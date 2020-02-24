from PyQt5 import QtWidgets, QtCore, QtGui

class MyLabel(QtWidgets.QWidget):
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setPen(QtCore.Qt.black)
        painter.translate(20, 100)
        painter.rotate(-90)
        painter.drawText(0, 0, "hellos")
        painter.end()