import numpy as np
import matplotlib.pyplot as plt

pi = np.pi

def gaussian(z0,sig):
  zz    = np.linspace(0.1,1.1,500)
  gauss = (1.0/np.sqrt(2.0*pi*(sig)))*np.exp(-0.5*(zz-z0)*(zz-z0)/sig/sig)
  """
  print (gauss/gauss.sum()).sum()
  plt.plot(zz,gauss,'k-')
  plt.show()
  """
  return gauss/gauss.sum()

def twogaussian(z0,dz,sig,sig1,ratio):
  zz     = np.linspace(0.1,1.1,500)
  amp    = ratio
  ztmp   = z0+dz
  gauss1 = amp*(1.0/np.sqrt(2.0*pi*(sig)))*np.exp(-0.5*(zz-z0)*(zz-z0)/sig/sig)
  gauss2 = (1.0/np.sqrt(2.0*pi*(sig)))*np.exp(-0.5*(zz-ztmp)*(zz-ztmp)/sig1/sig1)
  gauss  = (gauss1+gauss2)/(gauss1+gauss2).sum()
  """  
  plt.plot(zz,gauss,'k-',label='Total')
  #plt.plot(zz,gauss1,'r-',label='Gaussian 1')
  #plt.plot(zz,gauss2,'b-',label='Gaussian 2')
  plt.xlabel('z')
  plt.ylabel('P(z)')
  plt.legend()
  plt.savefig('mock_PofZ_1.eps')
  plt.show()
   
  plt.plot(zz,gauss)
  plt.xlabel('z')
  plt.ylabel('p(z)')
  plt.show()
  """
  return gauss
  
"""
def main():

  #gaussian(0.4,0.05)
  twogaussian(0.4,0.3,0.05,0.1,10.0)

if __name__=="__main__":
  main()
"""
