# 西瓜数据集3.0a

import numpy as np
import matplotlib.pyplot as plt
import math
# create dataset
def createDataSet():
#    x = np.array([[0.697, 0.460], [0.774, 0.376], [0.634, 0.264],
#                  [0.608, 0.318], [0.556, 0.215], [0.403, 0.237],
#                  [0.481, 0.149], [0.437, 0.211], [0.666, 0.091],
#                  [0.243, 0.267], [0.245, 0.057], [0.343, 0.099],
#                  [0.639, 0.161], [0.657, 0.198], [0.360, 0.370],
#                  [0.593, 0.042], [0.719, 0.103]])
#
#    y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#
#    plt.scatter(x[:,0],x[:,1], s=(y*30)+15,c=(y*20)+15)
#    plt.show()
    X1 = np.array(np.random.random((2,8))*5+4) # 类别A
    X2 = np.array(np.random.random((2,8))*5+3) # 类别B

    return X1, X2

def meanX(data):
    return np.mean(data, axis=1)    # axis=0表示按照行来求均值，如果输入list,则axis=1

# 计算类内离散度矩阵子项
def compute_si(xi):
    m, n = xi.shape
    ui = meanX(xi).reshape(-1,1)
    si = np.zeros((m, m))
    for i in range(0, n):
        si = si+np.dot((xi[:,i]-ui), (xi[:,i]-ui).T)
    return si/n

# 计算类间离散度矩阵子项
def compute_Sb(x1,x2):
    dataX = np.vstack((x1,x2))
    #print("dataX:" + str(dataX))
    # 计算均值
    u1 = meanX(x1).reshape(-1,1)
    u2 = meanX(x2).reshape(-1,1)
    u = meanX(dataX).reshape(-1,1)
    Sb = np.dot((u1-u2), (u1-u2).T)
    return Sb

def LDA(x1, x2):
    # 计算类内离散度矩阵
    s1 = compute_si(x1)
    s2 = compute_si(x2)
    Sw = s1 + s2

    # 计算类间离散度矩阵
    Sb = compute_Sb(x1, x2)

    # 求最大特征值对应的特征向量
    eig_value, vec = np.linalg.eig(np.mat(Sw).I*Sb) # 特征值和特征向量
    index_vec = np.argsort(-eig_value) # 对eig_value从大到小排序，返回索引
    eig_index = index_vec[:1] # 取出最大的特征值的索引
    w = vec[:, eig_index] # 取出最大的特征值对应的特征向量
    #w = vec
    return w

if __name__ == "__main__":
    X1, X2 = createDataSet()
    w = LDA(X1, X2)
    print("w=" + str(w))
    plt.scatter(X1[0, :], X1[1, :], s=np.zeros((X1.shape[1])) * 50 + 30, c='b', marker='*')
    plt.scatter(X2[0, :], X2[1, :], s=np.ones((X2.shape[1])) * 50 + 30, c='y')
    x = np.linspace(0,50,500)
    y1 = -w[0,0]/w[1,0]*x
    plt.plot(x,y1)
    z1 = abs(np.array(np.dot(w.transpose(), X1)))
    z2 = abs(np.array(np.dot(w.transpose(), X2)))
    theta = math.atan(-w[0,0]/w[1,0])
    print("theta=\t"+ str(theta))
    sin = math.sin(theta)
    cos = math.cos(theta)
    print(str(sin)+"\t"+str(cos))
    print(sin/cos-w[0,0]/w[1,0])
    #plt.scatter(z1*sin, z1*cos, s=np.zeros((X1.shape[1])) * 50 + 30, c='g', marker='*')
    #plt.scatter(z2*sin, z2*cos, s=np.zeros((X2.shape[1])) * 50 + 30, c='r')
    z1_x = list((z1*sin).reshape(-1))
    z1_y = list((z1*cos).reshape(-1))
    z2_x = list((z2*sin).reshape(-1))
    z2_y = list((z2*cos).reshape(-1))
    #for i in range(X1.shape[1]):
    #    plt.plot([X1[0, i], z1_x[i]], [X1[1,i], z1_y[i]], 'g--', lw=2)
    #for i in range(X2.shape[1]):
    #    plt.plot([X2[0, i], z2_x[i]], [X2[1,i], z2_y[i]], 'r--', lw=2)



    z = np.hstack((z1,z2))
    #plt.plot(x,y2)
    xmax = np.max(np.hstack((X2[0,:], X1[0,:], (z*cos).reshape(-1))))
    xmin = np.min(np.hstack((X2[0,:], X1[0,:], (z*cos).reshape(-1))))
    ymax = np.max(np.hstack((X2[1,:], X1[1,:], (z*sin).reshape(-1))))
    ymin = np.min(np.hstack((X2[1,:], X1[1,:], (z*sin).reshape(-1))))
    plt.xlim([xmin-1,xmax+1])
    plt.ylim([ymin-1,ymax+1])
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()
