import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pf

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
def pofz_hsc(indx):
   data = pf.getdata('mlz_photoz_pdf_stack4.fits')
   data1= pf.getdata('pozBins.fits')
   zbin = data1
	 
   p1   = data[0,:]
   p2   = data[1,:]
   p3   = data[2,:]
   p4   = data[3,:]
   if indx==1:
     pofz = p1
   if indx==2:
     pofz = p2
   if indx==3:
     pofz = p3
   if indx==4:
     pofz = p4
   struct={'zbin':zbin,'pofz':pofz}
   return struct
										     

def main():

  #gaussian(0.4,0.05)
  twogaussian(0.4,0.3,0.05,0.1,10.0)

if __name__=="__main__":
  main()
"""
