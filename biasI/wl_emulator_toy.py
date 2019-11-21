#!/home/wtluo/anaconda/bin/python2.7

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import posterior as post

# Basic Parameters---------------------------
#h = 0.673
h = 0.7
Om0 = 0.3
Ol0 = 1.0-Om0
Otot = Om0+Ol0
Ok0 = 1.0-Otot
w = -1
rho_crit0 = 2.78e11 #M_sun Mpc^-3 *h*h
rho_bar0 = rho_crit0*Om0
sigma8 = 0.8
apr =  206269.43                #1/1^{''}
vc = 2.9970e5                   #km/s
G = 4.3e-9                      #(Mpc/h)^1 (Msun/h)^-1 (km/s)^2 
H0 = 70.0                      #km/s/(Mpc/h)
pc = 3.085677e16                #m
kpc = 3.085677e19               #m
Mpc = 3.085677e22               #m
Msun = 1.98892e30               #kg
yr = 31536000.0/365.0*365.25    #second
Gyr = yr*1e9                    #second
pi  = np.pi
omega_m = Om0
omega_l = Ol0
omega_k = Ok0
Gg    =6.67408e-11              
ckm   =3.240779e-17            
ckg   =5.027854e-31           
fac   =(vc*vc*ckg)/(4.*pi*Gg*ckm) 
#resp,errorp = np.loadtxt('parents_dist.dat',unpack=True) 
zss   = np.array([0.2,0.4,0.6,0.8])
zs    = zss[3]
Nboot = 200
#--Basic cosmology calculations--------------------
def efunclcdm(x):
   res = 1.0/np.sqrt(Om0*(1.0+x)**3+Ok0*(1.0+x)**2+Ol0*(1.0+x)**(3*(1.0+w)))
   return res
def Hz(x):
   res = H0/efunclcdm(x)
   return res
#-----------------------------------------------------------------------------
def a(x):
   res = 1.0/(1.0+x)
   return res
#-----------------------------------------------------------------------------
def Dh():
   res = vc/H0
   return res
def Da(x):
   res = Dh()*integrate.romberg(efunclcdm, 0, x)
   return res

#--using NFW profile---- 
def funcs(Rp,rs):
  x   = Rp/rs
  x1  = x*x-1.0
  x2  = 1.0/np.sqrt(np.abs(1.0-x*x))
  x3  = np.sqrt(np.abs(1.0-x*x))
  x4  = np.log((1.0+x3)/(x))
  s1  = Rp*0.0
  s2  = Rp*0.0

  if x >0.0 and x<1.0:
    s1 = 1.0/x1*(1.0-x2*x4)
    s2 = 2.0/(x1+1.0)*(np.log(0.5*x)\
           +x2*x4)
  if x ==1.0:  
    x2 = x==1.0
    s1 = 1.0/3.0
    s2 = 2.0+2.0*np.log(0.5)

  if x>1.0:
    s1 = 1.0/x1*(1.0-x2*np.arctan(x3))
    s2 = 2.0/(x1+1.0)*(np.log(0.5*x)+\
             x2*np.arctan(x3))

  res = {'funcf':s1,'funcg':s2}
  return res
def nfwesd(theta,z,Rp):
  Mh,c      = theta
  efunc     = 1.0/np.sqrt(omega_m*(1.0+z)**3+\
              omega_l*(1.0+z)**(3*(1.0+w))+\
              omega_k*(1.0+z)**2)
  rhoc      = rho_crit0/efunc/efunc
  omegmz    = omega_m*(1.0+z)**3*efunc**2
  ov        = 1.0/omegmz-1.0
  dv        = 18.8*pi*pi*(1.0+0.4093*ov**0.9052)
  rhom      = rhoc*omegmz

  r200 = (10.0**Mh*3.0/200./rhom/pi)**(1./3.)
  rs   = r200/c
  delta= (200./3.0)*(c**3)\
          /(np.log(1.0+c)-c/(1.0+c))
  amp  = 2.0*rs*delta*rhoc*10e-14
  functions = funcs(Rp,rs)
  funcf     = functions['funcf']
  funcg     = functions['funcg']
  esd       = amp*(funcg-funcf)

  return esd

def Rpbins(theta,Nbin,z):
  Rmax  = 3.0
  Rmin  = 0.1
  dl    = Da(z)
  ds    = Da(zs)
  Sig   = fac*ds/(dl*(ds-dl))/(1.0+z)/(1.0+z)
  rbin  = np.zeros(Nbin+1)
  r     = np.zeros(Nbin)
  xtmp  = (np.log10(Rmax)-np.log10(Rmin))/Nbin
  area  = np.zeros(Nbin)
  ngals = np.zeros(Nbin)
  esd   = np.zeros(Nbin)
  esdnfw= np.zeros(Nbin)
  the1  = np.zeros(Nbin)
  the2  = np.zeros(Nbin)
  for i in range(Nbin):
    ytmp1 = np.log10(Rmin)+float(i)*xtmp
    ytmp2 = np.log10(Rmin)+float(i+1)*xtmp
    rbin[i] = 10.0**ytmp1
    rbin[i+1] = 10.0**ytmp2
    the1  = np.arctan(rbin[i]/dl)*180.0*60.0/pi
    the2  = np.arctan(rbin[i+1]/dl)*180.0*60.0/pi
    area[i]= 1.0*pi*(the2**2-the1**2)
    r[i] =(rbin[i])*0.5+(rbin[i+1])*0.5
    ngals[i]= np.random.poisson(lam=area[i])
    #ngals[i]= area[i]
    #if ngals[i]>0:
    esdnfw[i]= nfwesd(theta,z,r[i])/Sig
  return {'radius':r,'NUM':ngals,'NFW':esdnfw}

#-----marginalizing PofZ on ESDs---------------------------------
def esdsymmetric(theta):
   zz        = np.linspace(0.1,1.1,500) 
   ds        = np.zeros(500)
   for ids in range(500):
     ds[ids] = Da(zz[ids])

   zl,zs,zsig= theta
   dl        = Da(zl)

   gauss     = post.gaussian(zs,zsig)
   #plt.plot(zz,gauss,'k-',linewidth=3)
   #plt.xlabel('z')
   #plt.xlim(0.1,0.5)
   #plt.ylabel('P(z)')
   #plt.savefig('gauss.eps')
   #plt.show()
	  
   ix        = zz>=zl+0.01
   Sig_pofz  = (gauss[ix]*fac*ds[ix]/(dl*(ds[ix]-dl))/(1.0+zl)/(1.0+zl)).sum()
   return Sig_pofz 

def esdasymmetric(theta):
   zz        = np.linspace(0.1,1.1,500) 
   ds        = np.zeros(500)
   for ids in range(500):
     ds[ids] = Da(zz[ids])

   zl,zs,dz,zsig1,zsig2,ratio= theta
   dl        = Da(zl)
   ix        = zz>=zl+0.01
   pofz      = post.twogaussian(zs,dz,zsig1,zsig2,ratio)
   
   #plt.plot(zz,pofz,'k-',linewidth=3)
   #plt.xlabel('z')
   #plt.ylabel('p(z)')
   #plt.savefig('twogauss.eps')
   #plt.show()
      
   Sig_pofz  = (pofz[ix]*fac*ds[ix]/(dl*(ds[ix]-dl))/(1.0+zl)/(1.0+zl)).sum()
   
   return Sig_pofz 

def esdhsc(theta):
   zz        = np.linspace(0.1,1.1,500) 
   ds        = np.zeros(500)
   for ids in range(500):
     ds[ids] = Da(zz[ids])

   zl,zs,indx= theta

   dl        = Da(zl)
   ix        = zz>=zl+0.01
   struct    = post.pofz_hsc(indx)
   zbin      = struct['zbin']+dz
   tmp       = struct['pofz']
   tmp       = tmp/tmp.sum()
   ixx       = tmp ==np.max(tmp) 
   dz        = (zs-zbin[ixx])
   zbnew     = struct['zbin']+dz
   pofz      = np.interp(zz,zbnew,tmp)       
   Sig_pofz  = (pofz[ix]*fac*ds[ix]/(dl*(ds[ix]-dl))/(1.0+zl)/(1.0+zl)).sum()
   return Sig_pofz

#----------------------------------------------------------------
def main():

#------------------------------------------------------------------
   zl    = 0.1
   zs    = 0.3
   dl    = Da(zl)
   ds    = Da(zs)
   Nbin  = 30
   shear = np.zeros(Nbin)
   
   dz    = 0.2
   zsig1 = 0.05
   zsig2 = 0.1
   ratio = 2.0 

   Mh   = 14.0
   c    = 4.67*(10.0**(Mh-14)*h)**(-0.11) # Neto et al 2007 
   info = Rpbins([Mh,c],Nbin,zl)
   Rp   = info['radius']
   shear= info['NFW']
   Sig  =fac*ds/(dl*(ds-dl))/(1.0+zl)/(1.0+zl)
   esd_true = Sig*shear

   params1  = np.array([zl,zs,zsig1])
   esd_sym  = shear*esdsymmetric(params1)
   params2  = np.array([zl,zs,dz,zsig1,zsig2,ratio])
   esd_asym = shear*esdasymmetric(params2)

   #vm1  = np.mean(esd_sym/esd_true) 
   #vm2  = np.mean(esd_asym/esd_true) 
   #print vm2,vm1
   #---- Real test part ----------------------------------
   parab1   = np.array([zl,zs,1])
   esd_hsc1 = shear*esdhsc(parab1)
   parab2   = np.array([zl,zs,2])
   esd_hsc2 = shear*esdhsc(parab2)
   parab3   = np.array([zl,zs,3])
   esd_hsc3 = shear*esdhsc(parab3)
   parab4   = np.array([zl,zs,4])
   esd_hsc4 = shear*esdhsc(parab4)
   fig,axs=plt.subplots(nrows=2,ncols=1,sharex=True,
                   sharey=False,figsize=(8,8))
   l1,=axs[0].plot(Rp,esd_true,'k--',linewidth=3,label='True ESD')
   l2,=axs[0].plot(Rp,esd_sym,'b-.',linewidth=3,label='Gaussian pofz ESD')
   l3,=axs[0].plot(Rp,esd_asym,'g:',linewidth=3,label='twoGaussian pofz ESD')
   axs[0].plot(Rp,esd_asym,'g:',linewidth=3,label='twoGaussian pofz ESD')
   axs[0].set_xscale('log')
   axs[0].set_yscale('log',nonposy='clip')
   axs[0].set_ylabel('ESD ($M_{\odot}/pc^2)$',fontsize=20)
   axs[0].set_xlim(0.1,3.0)
   axs[0].set_ylim(1.001,500)
   axs[0].set_yticks([10,100])
   
   axs[1].plot(Rp,esd_true/esd_true,'k-',linewidth=3)
   axs[1].plot(Rp,esd_sym/esd_true,'b-.',linewidth=3)
   axs[1].plot(Rp,esd_asym/esd_true,'g:',linewidth=3)
   axs[1].set_xscale('log')
   axs[1].set_xlim(0.1,3.0)
   axs[1].set_ylim(0.9,1.05)
   axs[1].set_yticks([0.94,1.0,1.03])
   axs[1].set_ylabel('ratio (ESD_bias/ESD_true)',fontsize=20)
   fig.text(0.5,0.03,'R $h^{-1}Mpc$',ha='center',size=16)
   plt.subplots_adjust(wspace = 0.0, hspace = 0.0 )
   lines = [l1,l2,l3]
   axs[0].legend(lines,[l.get_label() for l in lines])
   plt.legend()
   plt.savefig('toy_bias.eps')
   plt.show()

if __name__=='__main__':
   main()
