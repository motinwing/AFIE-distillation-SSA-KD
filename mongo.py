import threading
import os

losses = []
accs = []


# 调用上面画图的那个draw_final.py程序
def draw_final():
    os.system("python .\\draw_final.py")


# 多线程，多出来的一条线程用于调用另一个程序
class myDraw_final(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    # 子线程用来调用那个py文件，此时，那个被调用的py文件中的程序运行在那个py文件的主线程里
    # 但是那个程序，是在我这个程序的子线程里被调用的
    def run(self):
        draw_final()


# UI窗口上的按钮与这个函数进行绑定，那个按钮按下后，创建子线程并开始执行上面那个类的run函数
def start_draw_final():
    my = myDraw_final()
    my.start()
