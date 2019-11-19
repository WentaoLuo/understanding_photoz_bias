"""
This is a code for simulating colored images.
"""

import galsim
import sys
import os
import numpy
import logging
import matplotlib.pyplot as plt

def chromatic_galaxy(theta):
  gal_type,half_radius,pixel_scale,skyvar,redshift,e1,e2,indx =theta 
  # where to find and output data
  path,filename = os.path.split(__file__)
  datapath      = os.path.abspath(os.path.join(path,"data/"))
  outpath       = os.path.abspath(os.path.join(path,"output/"))
  
  # In non-script code, use getLogger(__name__) at module scope instead
  logging.basicConfig(format="%(message)s",level=logging.INFO,stream=sys.stdout)
  logger        = logging.getLogger("colored_images")

  # Initialize (pseudo-) random number generator
  random_seed   = 1234567
  rng           = galsim.BaseDeviate(random_seed) 
  
  # read in SEDs
  SED_names     = ['CWW_E_ext','CWW_Sbc_ext','CWW_Scd_ext','CWW_Im_ext']
  SEDs          = {}
  for SED_name in SED_names:
      SED_filename   = os.path.join(datapath,'{1}.sed'.format(SED_name)) 
      SED            = galsim.SED(SED_filename,wave_type='Ang')
      SEDs[SED_name] = SED.withFluxDensity(target_flux_density=1.0,wavelength=500)
  logger.debug('Successfully read in SEDs') 

  filter_names = 'ugrizy'
  filters      = {}

  for filter_name in filter_names:
     filter_filename      = os.path.join(datapath,'LSST_{0}.dat'.format(filter_name))
     filters[filter_name] = galsim.Bandpass(filter_filename)
     filters[filter_name] = filters[filter_name].thin(rel_err=1e-4)

  logger.debug('Read in filters')
  PSF       = galsim.Moffat(fwhm=0.6,beta=2.5)

  #-------------------------------------------------------------
  # Part A: Chromatic de Vaucouleurs galaxy
  if gal_type == 'deVoucauleurs':
    logger.info('')
    logger.info('Starting part A: chromatic de Vaucouleurs galaxy')
    mono_gal  = galsim.DeVaucouleurs(half_light_radius=half_radius)
    plt.imshow(mono_gal.drawImage(64,64).array)
    plt.show()
    SED       = SEDs['CWW_E_ext'].atRedshift(redshift)
    gal       = galsim.Chromatic(mono_gal,SED)
  
    gal       = gal.shear(g1=0.5,g2=0.3).dilate(1.05).shift((1.0,2.1))
    logger.debug('Created Chromatic')

    final     = galsim.Convolve([gal,PSF])
    logger.debug('Created final profile')

    gaussian_noise = galsim.GaussianNoise(rng,sigma=skyvar)
    for filter_name,filter_ in filters.iteritems():
      img          = galsim.ImageF(64,64,scale=pixel_scale)
      final.drawImage(filter_,image=img)
      #plt.imshow(final.drawImage(64,64).array)
      #plt.show()
      #img.addNoise(gaussian_noise)
      logger.debug('Created {0}-band image'.format(filter_name))
      out_filename = os.path.join(outpath,'demo12a_{0}_{1}.fits'.format(filter_name,str(indx)))
      galsim.fits.write(img,out_filename)
      logger.debug('Wrote {0}-band image to disk'.format(filter_name))
      logger.info('Added flux for {0}-band image:{1}'.format(filter_name,img.added_flux))
  #-----------------------------------------------------------------------
  # PART B: chromatic bulge_disk galaxy
  if gal_type == 'diskbulge':
    logger.info('')
    logger.info('Starting part B: chromatic bulge_disk galaxy')

    mono_bulge = galsim.DeVaucouleurs(half_light_radius=0.05)
    bulge_SED  = SEDs['CWW_E_ext'].atRedshift(redshift)
    bulge      = mono_bulge * bulge_SED
    bulge      = bulge.shear(g1=0.05,g2=0.07)
    logger.debug('Created bulge component')
    mono_disk  = galsim.Exponential(half_light_radius=1.0)
    disk_SED   = SEDs['CWW_Im_ext'].atRedshift(redshift)
    disk       = mono_disk*disk_SED
    disk       = disk.shear(g1=e1,g2=e2)
    logger.debug('Created disk component')

    bdgal      = 1.1*(0.8*bulge+4.0*disk)
    bdfinal    = galsim.Convolve([bdgal,PSF])
    logger.debug('Created bulge+disk galaxy final profile')
    gaussian_noise = galsim.GaussianNoise(rng,sigma=skyvar)
    for filter_name,filter_ in filters.iteritems():
      img          = galsim.ImageF(64,64,scale=pixel_scale)
      bdfinal.drawImage(filter_,image=img)
      #img.addNoise(gaussian_noise)
      logger.debug('Created {0}-band image'.format(filter_name))
      out_filename = os.path.join(outpath,'demo12b_{0}_{1}.fits'.format(filter_name,str(indx)))
      galsim.fits.write(img,out_filename)
      logger.debug('Wrote {0}-band image to disk'.format(filter_name))
      logger.info('Added flux for {0}-band image:{1}'.format(filter_name,img.added_flux))
  # PART C: chromatic real galaxy
  if gal_type == 'real':
    logger.info('')
    logger.info('Starting part B: chromatic bulge_disk galaxy')
    cubeimg    = pf.getdata('cube_real.fits')
    idx        = np.random.randint(low=0,high=99)
    imarr      = cubeimg[idx,:,:]
    nx1,nx2    = np.shape(imarr) 
    img        = galsim.ImageF(nx1,nx2,scale=pixel_scale)
    bulge_SED  = SEDs['CWW_E_ext'].atRedshift(redshift)
    bulge      = mono_bulge * bulge_SED
    bulge      = bulge.shear(g1=0.05,g2=0.07)
    logger.debug('Created bulge component')
    mono_disk  = galsim.Exponential(half_light_radius=1.0)
    disk_SED   = SEDs['CWW_Im_ext'].atRedshift(redshift)
    disk       = mono_disk*disk_SED
    disk       = disk.shear(g1=e1,g2=e2)
    logger.debug('Created disk component')

    bdgal      = 1.1*(0.8*bulge+4.0*disk)
    bdfinal    = galsim.Convolve([bdgal,PSF])
    logger.debug('Created bulge+disk galaxy final profile')
    gaussian_noise = galsim.GaussianNoise(rng,sigma=skyvar)
    for filter_name,filter_ in filters.iteritems():
      img          = galsim.ImageF(64,64,scale=pixel_scale)
      bdfinal.drawImage(filter_,image=img)
      img.addNoise(gaussian_noise)
      logger.debug('Created {0}-band image'.format(filter_name))
      out_filename = os.path.join(outpath,'demo12b_{0}_{1}.fits'.format(filter_name,str(indx)))
      galsim.fits.write(img,out_filename)
      logger.debug('Wrote {0}-band image to disk'.format(filter_name))
      logger.info('Added flux for {0}-band image:{1}'.format(filter_name,img.added_flux))
       
#-----------------------------------------------------------------------       
def main():
  #gal_type,half_radius,pixel_scale,skyvar,redshift,e1,e2 
  theta = ['deVoucauleurs',0.8,0.2,0.0,0.01,0.4,0.2,0]
  chromatic_galaxy(theta)
  #theta = ['diskbulge',0.5,0.2,0.0,0.01,-0.8,0.2,0]
  #chromatic_galaxy(theta)
  #theta = ['deVoucauleurs',0.8,0.2,0.0,0.01,0.4,-0.2,1]
  #chromatic_galaxy(theta)
  #theta = ['diskbulge',0.5,0.2,0.0,0.01,0.6,-0.2,1]
  #chromatic_galaxy(theta)
  #theta = ['deVoucauleurs',0.8,0.2,0.0,0.01,0.4,0.2,2]
  #chromatic_galaxy(theta)
  #theta = ['diskbulge',0.5,0.2,0.0,0.01,0.6,0.2,2]
  #chromatic_galaxy(theta)
  #theta = ['deVoucauleurs',0.8,0.2,0.0,0.01,-0.4,0.2,3]
  #chromatic_galaxy(theta)
  #theta = ['diskbulge',0.5,0.2,0.0,0.01,00.6,0.2,3]
  #chromatic_galaxy(theta)
if __name__ == '__main__':
   main()
