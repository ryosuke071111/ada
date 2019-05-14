import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
np.random.seed(0)
ans=10**9
ans_h=10**9
ans_l=10**9



#データを生成する
def generate_sample(xmin=-3,xmax=3,sample_size=50):
  x=np.linspace(start=xmin,stop=xmax,num=sample_size)
  pix=np.pi*x
  target=np.sin(pix)/pix+0.1*x
  noise=0.05*np.random.normal(loc= 0.,scale=1.,size=sample_size)
  return x,target+noise

#ガウスカーネル
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

def cross_validation(x,y,n,h,lam,i):
  index=x.size//n
  tmp=0
  x,y=shuffle(x,y)

  for i in range(n):
    #データ分割
    x_test=np.hstack((x[:index*i],x[index*(i+1):]))
    x_train=np.array(x[i*index:(i+1)*index])
    y_test=np.hstack((y[:index*i],y[index*(i+1):]))
    y_train=y[i*index:(i+1)*index]
    #予測値&誤差を返す
    idx=np.argsort(x_test)
    x_test=x_test[idx]
    y_test=y_test[idx]
    prediction=calc_design_matrix(x_train,x_test,y_train,h,lam)
    truth=calc_design_matrix(x_test,x_test,y_test,1,0)
    error=sqared_error(y_test,prediction)
    print(" fold-",i,":",error)
    tmp+=error
  ax=fig.add_subplot(3,3,j)
  ax.set_title("lambda:{} h:{}".format(lam,h))
  ax.scatter(x_train,y_train,s=6)
  ax.plot(x_test,prediction,c='red')
  ax.plot(x_test,truth,c='green')

  print(" *****結果*****;",tmp/n)
  return tmp/n

x, y = generate_sample(sample_size=1000)
H=[0.1,1.0,100.0]
L=[0.1,1.0,100.0]

axes=[]
fig=plt.figure(figsize=(9,6))

fig.subplots_adjust(hspace=0.6, wspace=0.6)
j=1
for h in H:
  for l in L:
    print("h-value:",h)
    print("l-value:",l)
    tmp=cross_validation(x,y,10,h,l,j)
    if ans>tmp:
      ans=tmp
      ans_l=l
      ans_h=h
    j+=1
print('最適解は h:',ans_h,"l:",ans_l,"誤差:",ans)
plt.show()

