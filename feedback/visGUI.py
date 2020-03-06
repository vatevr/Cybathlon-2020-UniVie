import sys
from PyQt5 import QtWidgets, QtCore, QtGui
import time
from random import randrange
from pylsl import StreamInlet, resolve_stream
import datetime

UPDATE_INTERVAL = 100  # UPDATING INTERVAL (ms)
STREAM_TYPE = 'VIS'  # STREAM TYPE

MIN_VALUE = 0
MAX_VALUE = 100

class Window(QtWidgets.QMainWindow):

    def __init__(self):
        super(Window, self).__init__()
        self.setGeometry(500, 100, 800, 800)
        self.setStyleSheet("background-color: lightgrey;")

        self.initLabels()
        self.initGeo()

        self.timerSignalOV = QtCore.QTimer()
        self.timerSignalOV.setInterval(UPDATE_INTERVAL)
        self.timerSignalOV.timeout.connect(self.updateSignal)

        self.timerTime = QtCore.QTimer()
        self.timerTime.setInterval(25)
        self.timerTime.timeout.connect(self.updateTime)

        self.currentTimer = self.timerSignalOV

        QtWidgets.QMessageBox.information(self, 'Vis Info',
                                          "<strong style='font-size: 15px;'>Start your scenario...</strong>")
        print("looking for an EEG stream...")

        self.streams = resolve_stream('type', STREAM_TYPE) # STREAM TYPE (must match the type from "LSL Export (Gipsa)" box in OV)
        # create a new inlet to read from the stream
        self.inlet = StreamInlet(self.streams[0])

    def initLabels(self):
        self.t1 = QtWidgets.QLabel(self)
        self.t1.setText("HEADLIGHTS")
        self.t1.setGeometry(330, 0, 140, 50)
        self.t1.setStyleSheet("font-weight: bold; font-size: 15pt;")
        self.t1.setAlignment(QtCore.Qt.AlignCenter)

        self.t2 = QtWidgets.QLabel(self)
        self.t2.setText("LEFT")
        self.t2.setGeometry(50, 330, 80, 50)
        self.t2.setStyleSheet("font-weight: bold; font-size: 15pt;")
        self.t2.setAlignment(QtCore.Qt.AlignLeft)

        self.t3 = QtWidgets.QLabel(self)
        self.t3.setText("RIGHT")
        self.t3.setGeometry(670, 330, 80, 50)
        self.t3.setStyleSheet("font-weight: bold; font-size: 15pt;")
        self.t3.setAlignment(QtCore.Qt.AlignRight)

        self.t4 = QtWidgets.QLabel(self)
        self.t4.setText("REST")
        self.t4.setGeometry(330, 750, 140, 50)
        self.t4.setStyleSheet("font-weight: bold; font-size: 15pt;")
        self.t4.setAlignment(QtCore.Qt.AlignCenter)

    def initGeo(self):
        # Bars
        self.progressLeft = QtWidgets.QProgressBar(self)
        self.progressLeft.setGeometry(50, 360, 300, 80)
        self.progressLeft.setInvertedAppearance(True)
        self.progressLeft.setStyleSheet("QProgressBar::chunk { background-color: red; }")
        self.progressLeft.setOrientation(QtCore.Qt.Horizontal)
        self.progressLeft.setMinimum(MIN_VALUE)
        self.progressLeft.setMaximum(MAX_VALUE)
        self.progressLeft.setValue(40)
        self.progressLeft.setTextVisible(False)

        self.progressLight = QtWidgets.QProgressBar(self)
        self.progressLight.setGeometry(360, 50, 80, 300)
        self.progressLight.setStyleSheet("QProgressBar::chunk { background-color: red; }")
        self.progressLight.setOrientation(QtCore.Qt.Vertical)
        self.progressLight.setMinimum(MIN_VALUE)
        self.progressLight.setMaximum(MAX_VALUE)
        self.progressLight.setValue(40)
        self.progressLight.setTextVisible(False)

        self.progressRest = QtWidgets.QProgressBar(self)
        self.progressRest.setGeometry(360, 450, 80, 300)
        self.progressRest.setStyleSheet("QProgressBar::chunk { background-color: red; }")
        self.progressRest.setOrientation(QtCore.Qt.Vertical)
        self.progressRest.setInvertedAppearance(True)
        self.progressRest.setMinimum(MIN_VALUE)
        self.progressRest.setMaximum(MAX_VALUE)
        self.progressRest.setValue(40)
        self.progressRest.setTextVisible(False)

        self.progressRight = QtWidgets.QProgressBar(self)
        self.progressRight.setGeometry(450, 360, 300, 80)
        # centring text works, but still do no rotate it
        self.progressRight.setStyleSheet("QProgressBar::chunk { background-color: red; }")
        # border-top-right-radius: 5px; border-bottom-right-radius: 5px;
        self.progressRight.setOrientation(QtCore.Qt.Horizontal)
        self.progressRight.setMinimum(MIN_VALUE)
        self.progressRight.setMaximum(MAX_VALUE)
        self.progressRight.setValue(40)
        self.progressRight.setTextVisible(False)

        # Buttons
        self.buttonstart = QtWidgets.QPushButton('Start', self)
        self.buttonstart.setGeometry(50, 50, 80, 50)
        self.buttonstart.clicked.connect(self.startT)

        self.buttonstop = QtWidgets.QPushButton('Stop', self)
        self.buttonstop.setGeometry(150, 50, 80, 50)
        self.buttonstop.clicked.connect(self.stopT)

        self.buttonDemoOV = QtWidgets.QPushButton('OpenViBE', self)
        self.buttonDemoOV.setGeometry(50, 110, 180, 30)
        self.buttonDemoOV.clicked.connect(self.changeDemoOV)

        self.buttonDemoTime = QtWidgets.QPushButton('Time demo (milliseconds)', self)
        self.buttonDemoTime.setGeometry(50, 150, 180, 30)
        self.buttonDemoTime.clicked.connect(self.changeDemoTime)

        self.buttonexit = QtWidgets.QPushButton('Close', self)
        self.buttonexit.setGeometry(50, 190, 180, 30)
        self.buttonexit.clicked.connect(self.close)



        # Plots (IN PROGRESS)
        '''
        self.graphWidget = pg.PlotWidget()
        self.graphWidget1 = QtWidgets.QWidget(self.graphWidget)
        self.graphWidget1.setGeometry(500, 190, 180, 30)

        hour = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        temperature = [30, 32, 34, 32, 33, 31, 29, 32, 35, 45]

        # plot data: x, y values
        self.graphWidget.plot(hour, temperature)
        '''

        # self.button.clicked.connect(self.updateOne)

        '''
        progress4 = MyBar("myServer008-Load", 10, 8, self)
        progress4.setGeometry(200, 50, 25, 150)
        '''

    def update(self):
        count = 0
        inc = True
        while True:
            if inc:
                while count < 10:
                    count += 1
                    time.sleep(0.3)
                    self.progressLeft.setValue(count)
                inc = False
            else:
                while count > 0:
                    count -= 1
                    time.sleep(0.3)
                    self.progressLeft.setValue(count)
                inc = True

    def startT(self):
        self.currentTimer.start()

    def stopT(self):
        self.currentTimer.stop()

    def progressColor(self, x):
        if x <= 70:
            return "QProgressBar::chunk { background-color: rgb(200, " + str(int(200 * (x / 70))) + " ,0 ) }"
        else:
            return "QProgressBar::chunk { background-color: rgb(" + \
                   str(int(200 * (1 - ((x - 70) / 30)))) + ",200 ,0 ) }"

    def update_bars(self):
        temp = randrange(100)
        self.progressLeft.setValue(temp)
        self.progressLeft.setStyleSheet(self.progressColor(temp))

    def updateSignal(self):
        sample, timestamp = self.inlet.pull_sample()
        print(self.inlet.pull_sample())
        sample = [ (x * 100) for x in sample]
        # print(sample, val)
        self.progressLeft.setStyleSheet(self.progressColor(sample[0]))
        self.progressLeft.setValue(sample[0])

        self.progressRight.setStyleSheet(self.progressColor(sample[3]))
        self.progressRight.setValue(sample[1])

        self.progressLight.setStyleSheet(self.progressColor(sample[1]))
        self.progressLight.setValue(sample[2])

        self.progressRest.setStyleSheet(self.progressColor(sample[2]))
        self.progressRest.setValue(sample[3])



    def updateTime(self):
        sample = [int(datetime.datetime.now().microsecond / 10000), int(datetime.datetime.now().microsecond / 10000), int(datetime.datetime.now().microsecond / 10000), int(datetime.datetime.now().microsecond / 10000)]
        self.progressLeft.setStyleSheet(self.progressColor(sample[0]))
        self.progressLeft.setValue(sample[0])

        self.progressLight.setStyleSheet(self.progressColor(sample[1]))
        self.progressLight.setValue(sample[1])

        self.progressRest.setStyleSheet(self.progressColor(sample[2]))
        self.progressRest.setValue(sample[2])

        self.progressRight.setStyleSheet(self.progressColor(sample[3]))
        self.progressRight.setValue(sample[3])

    def changeDemoOV(self):
        self.stopT()
        self.currentTimer = self.timerSignalOV

    def changeDemoTime(self):
        self.stopT()
        self.currentTimer = self.timerTime

    def close(self):
        sys.exit("Exit by user")

    def closeEvent(self, event):

        quit_msg = "Are you sure you want to exit the program?"
        reply = QtWidgets.QMessageBox.question(self, 'Message',
                                           quit_msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()



app = QtWidgets.QApplication(sys.argv)

GUI = Window()
GUI.show()

sys.exit(app.exec_())
