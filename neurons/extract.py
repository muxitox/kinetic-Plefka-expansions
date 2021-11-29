#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy.io
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
import os

dataset=1
filename='data/DataSet'+str(dataset)+'.mat'
mat = scipy.io.loadmat(filename)

print(mat.keys())
data=mat['data']
spikes=data[0][0][0]

N=len(spikes)

r=70
T=int(np.ceil(3600000/r))
X=csr_matrix((N,T))
for n in range(N):
	ts = spikes[n][0][0]
	print(len(ts),min(ts),max(ts))
	inds = np.floor(ts/r).astype(int)
	X[n,inds]=1

#print(X[0,:])
#plt.figure()
#plt.plot(X[0,:])
#plt.show()

#X = X*2-1

X=X[:,500:]

S=X.sum(axis=1)
S=np.asarray(S).reshape(-1).astype(int)
print(sorted(S))
print(min(S))

thS=60*10
thS = 0
print(len(S),sum(S>thS))

N=sum(S>thS)


X=X[S>thS,:]


k=X.mean(axis=0)
k=np.asarray(k).reshape(-1)

plt.figure()
plt.plot(k*N)

W=1000
w=np.ones(W)/W
plt.figure()
plt.plot(np.convolve(k*N,w))
plt.show()

th=8

sig=(k*N)>=th
#Xp=X[:,sig[0:-1]]
#X=X[:,sig[1:]]

d=1
Xd = X[:,0:-d]
X = X[:,d:]

#sig=sig[0:-d]
#X1d=Xd[:,sig]
#X1=X[:,sig]
#X0d=Xd[:,np.logical_not(sig)]
#X0=X[:,np.logical_not(sig)]
print(np.median(k)*N)
print(np.mean(k)*N)


plt.figure()
plt.hist(k*N,20)

plt.figure()
plt.hist(k**2*N,20)

#plt.show()
#exit()

m=X.mean(axis=1)
m=np.asarray(m).reshape(-1)

#mu1=np.asarray(X1.mean(axis=1)).reshape(-1) 
#mu0=np.asarray(X0.mean(axis=1)).reshape(-1) 
#l1=np.asarray(X1d.mean(axis=1)).reshape(-1) 
#l0=np.asarray(X0d.mean(axis=1)).reshape(-1) 

#plt.figure()
#plt.plot(mu1)
#plt.figure()
#plt.plot(mu0)
#plt.figure()
#plt.plot(l1)
#plt.figure()
#plt.plot(l0)
#plt.figure()
#plt.plot(m)
#plt.show()



dk=k-np.mean(m)

#P1=np.mean(sig)
#P0=1-P1

#T1=X1.shape[1]
#T0=X0.shape[1]

#print(T,T1,T0)

#plt.figure()
#plt.plot(dk[0:-1]*dk[1:])


#plt.show()
plt.figure()
plt.hist(m,40)


iu1 = np.triu_indices(N, 1)
C = X.dot(X.T).todense()/T
#Cl0 = X0d.dot(X0d.T).todense()/T0
#Cl1 = X1d.dot(X1d.T).todense()/T1

print(np.max(X))#,np.max(X0),np.max(X1))
print(np.max(C))#,np.max(Cl0),np.max(Cl1))
#exit()

D = X[:,0:-d].dot(X[:,d:].T).todense()/T
#Dmu1 = X1.dot(X1d.T).todense()/T1
#Dmu0 = X0.dot(X0d.T).todense()/T0

C-=np.einsum('i,k->ik',m,m, optimize=True)
D-=np.einsum('i,k->ik',m,m, optimize=True)

#Cl0-=np.einsum('i,k->ik',l0,l0, optimize=True)
#Cl1-=np.einsum('i,k->ik',l1,l1, optimize=True)

#Dmu1-=np.einsum('i,k->ik',mu1,l1, optimize=True)
#Dmu0-=np.einsum('i,k->ik',mu0,l0, optimize=True)

C*=4
D*=4
#Cl0*=4
#Cl1*=4
#Dmu1*=4
#Dmu0*=4


m = m*2-1
#mu0 =mu0*2-1
#mu1 =mu1*2-1
#l0 =l0*2-1
#l1 =l1*2-1
#print(C.shape)
del X
#print(C.shape)
#print(D.shape)

C[range(N),range(N)] = 1 - m**2

folder = 'stats'
isExist = os.path.exists(folder)
if not isExist:
	# Create a new directory because it does not exist
	os.makedirs(folder)

filename = folder + '/stats'+str(dataset)+'.npz'
np.savez_compressed(filename,m=m, C=C,D=D,r=r,N=N,T=T)



#plt.figure()
#plt.hist(Cl1[iu1],40)


plt.figure()
plt.imshow(C,aspect='auto')
plt.colorbar()


#plt.figure()
#plt.imshow(Cl1,aspect='auto')
#plt.colorbar()


#plt.figure()
#plt.imshow(Cl0,aspect='auto')
#plt.colorbar()



plt.figure()
plt.hist(np.asarray(D).reshape(-1),40)

#plt.figure()
#plt.hist(np.asarray(Dmu1).reshape(-1),40)

#plt.figure()
#plt.hist(np.asarray(Dmu0).reshape(-1),40)

plt.figure()
plt.imshow(D,aspect='auto')
plt.colorbar()


#plt.figure()
#plt.imshow(Dmu1,aspect='auto')
#plt.colorbar()

#plt.figure()
#plt.imshow(Dmu0,aspect='auto')
#plt.colorbar()

plt.show()



#plt.figure()
#plt.hist(C[iu1],40)
#plt.figure()
#plt.imshow(C,aspect='auto')
#plt.colorbar()

#plt.figure()
#plt.hist(C[iu1],40)
#plt.figure()
#plt.hist(np.asarray(D).reshape(-1),40)


#plt.figure()
#plt.imshow(D,aspect='auto')
#plt.colorbar()

#plt.show()
#C = np.einsum('it,kt->ik',X,X, optimize=True)/T -  np.einsum('i,k->ik',m,m, optimize=True)
#D = np.einsum('it,kt->ik',X[:,0:-1],X[:,1:], optimize=True)/T -  np.einsum('i,k->ik',m,m, optimize=True)


#plt.figure()
#plt.imshow(X,aspect='auto')
plt.show()

