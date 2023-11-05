import os
#https://matplotlib.org/stable/api/backend_qt_api.html
#os.environ['QT_API'] = 'PyQt6'
import matplotlib
matplotlib.use('QtAgg')
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')
plt.show()