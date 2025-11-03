import matplotlib.pyplot as plt
import csv


# 对ori数据进行查询
def find_final():
    plt.ion()
    while 1:
        accs = []
        with open(".\\data.csv", "r") as f:
            reader = csv.reader(f)
            for row in reader:
                accs.append(float(row[0]))
            f.close()

        plt.clf()  # 清除之前画的图
        plt.plot(range(len(accs)), accs)
        plt.pause(10)
        plt.ioff()  # 关闭画图窗口


if __name__ == "__main__":
    find_final()
