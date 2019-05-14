import numpy as np
import matplotlib
from collections import Counter
import pandas as pd
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
path = "digit_data/"


h=1.0
lam=0.1

#2
# trainデータ作成
def calc_design_matrix(x_train,x_test,y_train,h,lam):
  global theta
  #thetaを算出
  k=np.exp(-(x_train[None]-x_train[:,None])**2/(2*h**2))
  theta=np.linalg.solve(k.T.dot(k)+lam*np.identity(len(k)),k.T.dot(y_train[:,None]))
  #データからカーネル関数設計
  K = np.exp(-(x_train[None]-x_test[:,None])**2/(2*h**2))
  return K.dot(theta) #予測値を返す

#二乗誤差
def sqared_error(y,t):
  y=y.flatten()
  t=t.flatten()
  return np.sum(((y-t)**2))

def build_design_mat(x1, x2, bandwidth):
    return np.exp(
        -np.sum((x1[:, None] - x2[None]) ** 2, axis=-1) / (2 * bandwidth ** 2))

def optimize_param(design_mat, y, regularizer):
    return np.linalg.solve(
        design_mat.T.dot(design_mat) + regularizer * np.identity(len(y)),
        design_mat.T.dot(y))


def predict(train_data, test_data, theta):
    return build_design_mat(train_data, test_data, 10.).T.dot(theta)


def build_confusion_matrix(train_data, data, theta):
    confusion_matrix = np.zeros((10, 10), dtype=np.int64)
    for i in range(10):
        test_data = np.transpose(data[:, :, i], (1, 0))
        prediction = predict(train_data, test_data, theta)
        confusion_matrix[i][0] = np.sum(
            np.where(prediction > 0, 1, 0))
        confusion_matrix[i][1] = np.sum(
            np.where(prediction < 0, 1, 0))
    return confusion_matrix

def generate_data():
  x=np.array(pd.read_csv(path+"digit_train0.csv",delimiter=","))
  for i in range(1,10):
    a=np.array(pd.read_csv(path+"digit_train"+str(i)+".csv",delimiter=",")).reshape((499,256))
    x=np.vstack((x,a))
  y = np.array([[i]*499 for i in range(1,11)]).ravel()
  return x, y

x, y = generate_data()
design_mat = build_design_mat(x, x, 10.)
exit()


theta = optimize_param(design_mat, y, 1.)

confusion_matrix = build_confusion_matrix(x, x, theta)
print('confusion matrix:')
print(confusion_matrix)

exit()


#訓練データからパラメタ取得


# testデータ作成
T=np.array(pd.read_csv(path+"digit_test0.csv",delimiter=","))
T=T.reshape((199,256,1))
for i in range(1,10):
  a=np.array(pd.read_csv(path+"digit_test"+str(i)+".csv",delimiter=",")).reshape((199,256,1))
  T=np.dstack((T,a))

#テストデータを一つずつ見ていく
df={}
for i in range(10):
  t=T[:,:,i].T #i+1の数字が出る
  invs=np.linalg.inv(s+0.000001*np.eye(256))
  p1=(((mu1).dot(invs)).dot(t)-(((mu1).dot(invs)).dot(mu1)/2))
  p2=(((mu2).dot(invs)).dot(t)-(((mu2).dot(invs)).dot(mu2)/2))
  p3=(((mu3).dot(invs)).dot(t)-(((mu3).dot(invs)).dot(mu3)/2))
  p4=(((mu4).dot(invs)).dot(t)-(((mu4).dot(invs)).dot(mu4)/2))
  p5=(((mu5).dot(invs)).dot(t)-(((mu5).dot(invs)).dot(mu5)/2))
  p6=(((mu6).dot(invs)).dot(t)-(((mu6).dot(invs)).dot(mu6)/2))
  p7=(((mu7).dot(invs)).dot(t)-(((mu7).dot(invs)).dot(mu7)/2))
  p8=(((mu8).dot(invs)).dot(t)-(((mu8).dot(invs)).dot(mu8)/2))
  p9=(((mu9).dot(invs)).dot(t)-(((mu9).dot(invs)).dot(mu9)/2))
  p10=(((mu10).dot(invs)).dot(t)-(((mu10).dot(invs)).dot(mu10)/2))
  p=np.vstack((p1,p2))
  p=np.vstack((p,p3))
  p=np.vstack((p,p4))
  p=np.vstack((p,p5))
  p=np.vstack((p,p6))
  p=np.vstack((p,p7))
  p=np.vstack((p,p8))
  p=np.vstack((p,p9))
  p=np.vstack((p,p10))
  tmp={i:0 for i in range(1,11)}
  ls=Counter(p.argmax(axis=0))
  for k,v in ls.items():
    tmp[k+1]=v
  df[i+1]=list(tmp.values())
df=pd.DataFrame(df,index=[1,2,3,4,5,6,7,8,9,10])
print(df)




