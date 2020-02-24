from PyQt5 import QtWidgets, QtCore, QtGui

class MyBar(QtWidgets.QWidget):
    # Creates custom 'vertical progress bar'

    def __init__(self, text, maximumValue, currentValue, parent=None):
        super(MyBar, self).__init__(parent)
        self.text = text
        self.maximumValue = maximumValue
        self.currentValue = currentValue

    def setValue(self, currentValue):
        if self.currentValue != currentValue:
            self.currentValue = currentValue
            self.repaint()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.translate(0, self.height() - 1)
        painter.rotate(-90)

        painter.setPen(QtGui.QColor(140, 138, 135))
        painter.drawRoundedRect(QtCore.QRectF(0, 0, self.height() - 1, self.width() - 1), 4, 4)

        painter.setPen(QtGui.QColor(201, 199, 197))
        path = QtGui.QPainterPath()
        path.addRoundedRect(QtCore.QRectF(1, 1, self.height() - 3, self.width() - 3), 3, 3)

        painter.fillPath(path, QtGui.QColor(214, 212, 210))
        painter.drawPath(path)

        path = QtGui.QPainterPath()
        path.addRoundedRect(
            QtCore.QRectF(1, 1, (self.height() - 3) * self.currentValue / self.maximumValue, self.width() - 3), 3,
            3)

        painter.fillPath(path, QtGui.QColor(255, 0, 0))
        painter.drawPath(path)

        painter.setPen(QtCore.Qt.black)
        # print text centered
        painter.drawText(5, self.width() / 2 + 5, self.text)
        painter.end()