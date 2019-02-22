from PyQt5.QtWidgets import QLabel

class QLabel_Pixmap(QLabel):
    def __init__(self, parent=None):
        super(QLabel, self).__init__(parent)
        self.interface = None
    
    def resizeEvent(self,event):
        if self.interface and self.interface.is_pixmap:
            self.interface.draw_frame()
        else:
            super(QLabel, self).resizeEvent(event)