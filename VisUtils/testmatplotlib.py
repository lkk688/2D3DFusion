import os
#https://matplotlib.org/stable/api/backend_qt_api.html
print(os.environ.get('QT_API'))
#os.environ['QT_API'] = 'pyqt5'
import matplotlib
matplotlib.use('QtAgg')
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')
plt.show()