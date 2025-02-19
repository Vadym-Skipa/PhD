import numpy as np
from pylab import *
import h5py
import datetime
import matplotlib.pyplot as plt

a = r"/home/vadymskipa/PhD_student/data/processed_data_for_receivers/2024/"

fname='site_130_2024.h5'


f = h5py.File(a + fname)

ct=f['data']

site_ind_base=ct['site']==b'base'
site_ind_bas2=ct['site']==b'bas2'
site_ind_bas3=ct['site']==b'bas3'


ct_st_base=ct[site_ind_base]
ct_st_bas2=ct[site_ind_bas2]
ct_st_bas3=ct[site_ind_bas3]


vtec_base=ct_st_base['vtec']
utdoy_base=ct_st_base['time']
elev_base=ct_st_base['el']
az_base=ct_st_base['az']

vtec_bas2=ct_st_bas2['vtec']
utdoy_bas2=ct_st_bas2['time']


vtec_bas3=ct_st_bas3['vtec']
utdoy_bas3=ct_st_bas3['time']


figure(figsize=(8,11))

ax1=subplot(4,1,1)
ax1.plot(utdoy_base,vtec_base,'.')
ax1.set_xticklabels([''])
ylabel('TECu')
title_str='VTEC, base, %i, 2024'%(int(utdoy_base[0]+1))
title(title_str)
ylim([0,80])
#xlim([35,36])
grid()


ax2=subplot(4,1,2)
ax2.plot(utdoy_bas2,vtec_bas2,'.')
ax2.set_xticklabels([''])
ylabel('TECu')
title_str='VTEC, bas2, %i, 2024'%(int(utdoy_base[0]+1))
title(title_str)
ylim([0,80])
#xlim([35,36])
grid()

ax3=subplot(4,1,3)
ax3.plot(utdoy_bas3,vtec_bas3,'.')
#ax3.set_xticklabels([''])
ylabel('TECu')
title_str='VTEC, bas3, %i, 2024'%(int(utdoy_base[0]+1))
title(title_str)
ylim([0,80])
#xlim([35,36])
grid()

xlabel('UT DOY')

fstr='ftec_2024_%03i_base_bas2_bas3.png'%(int(utdoy_base[0]+1))
savefig(a + fstr)
