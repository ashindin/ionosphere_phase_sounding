import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import signal
from scipy import constants
from scipy import io
import sys
import time
import struct
from subprocess import check_output

t = time.time()

wv_filename="04061417_part01.wv"
z_filename= "doppler_shifts.z"
mat_filename="phase.mat"
n_ch=4 # number of channels
fd=1000000 # sample frequency (Hz)
period_sec=0.1 # interpulse period (seconds)
num_pulses=3300 # number of pulses
offset_bytes=41151200 
f_min=-425000 # low start frequency (Hz)
f_bandwidth=850000 # frequency bandwidth (Hz)
f_center=5400000
f_step=1000 # frequency step (Hz)
l_c_freq=100 # low cutoff frequency (Hz)
f_order=3 # filter order

period=period_sec*fd
scale_factor=0.5/32768
win=4000 # window size in samples
win2=1000 # window2 size in samples
del_rw=2550 # delay of the reflected signal in samples

w_axe1=np.arange(f_min+fd,fd,f_step)*2*constants.pi/fd
w_axe2=np.arange(0,-f_min+f_step,f_step)*2*constants.pi/fd
w_axe=np.concatenate((w_axe1,w_axe2), axis=0)

strange_shift=(-2*(offset_bytes/8+win)+period-del_rw+win) # strange time shift

sos_coefs = signal.iirfilter(f_order, (2**.5)*2*l_c_freq/fd, btype='lowpass', analog=False, ftype='bessel', output='sos')
sos_zi=np.zeros((2,2))

gwr_mat=np.zeros((num_pulses, win2))
gwi_mat=np.zeros((num_pulses, win2))
rwr_mat=np.zeros((num_pulses, win2))
rwi_mat=np.zeros((num_pulses, win2))
gw_ind_mat=np.zeros((num_pulses, win2))
rw_ind_mat=np.zeros((num_pulses, win2))

gw_amp_mat=np.zeros((len(w_axe),num_pulses))
rw_amp_mat=np.zeros((len(w_axe),num_pulses))
gw_pha_mat=np.zeros((len(w_axe),num_pulses))
rw_pha_mat=np.zeros((len(w_axe),num_pulses))

until_par="--until=+" + str(win)
unpack_str="<"+str(n_ch*win)+"h"

for pulse_counter in range(0,num_pulses):
    skip_val=int(offset_bytes/2/n_ch)+period*(pulse_counter)
    
    #ind_axe_start=skip_val+strange_shift
	ind_axe_start=skip_val
    ind_axe_all=np.arange(ind_axe_start,ind_axe_start+win) # samples axe for 4000-window
        
    skip_str="--skip="+str(skip_val)
    data_bytes=check_output(["wvunpack", "-r", "-q", skip_str, until_par, wv_filename,"-o", "-"])
    data=struct.unpack(unpack_str,data_bytes)
    
    data_scaled=np.reshape(data,[len(data)/n_ch, n_ch])*scale_factor
        
    gwOr_data=data_scaled[0:win2,0] # O-mode ground wave's real data
    gwOi_data=data_scaled[0:win2,1] # O-mode ground wave's imagenary data
    gwXr_data=data_scaled[0:win2,2] # X-mode ground wave's real data
    gwXi_data=data_scaled[0:win2,3] # X-mode ground wave's imagenary
        
    rwOr_data=data_scaled[del_rw:del_rw+win2,0] # O-mode reflected wave's real data
    rwOi_data=data_scaled[del_rw:del_rw+win2,1] # O-mode reflected wave's imagenary data        
    rwXr_data=data_scaled[del_rw:del_rw+win2,2] # X-mode reflected wave's real data
    rwXi_data=data_scaled[del_rw:del_rw+win2,3] # X-mode reflected wave's imagenary data        
        
    gwr_mat[pulse_counter,:]=gwOr_data+gwXi_data
    gwi_mat[pulse_counter,:]=gwOi_data-gwXr_data
    rwr_mat[pulse_counter,:]=rwOr_data+rwXi_data
    rwi_mat[pulse_counter,:]=rwOi_data-rwXr_data
    gw_ind_mat[pulse_counter,:]=ind_axe_all[0:win2]
    rw_ind_mat[pulse_counter,:]=ind_axe_all[del_rw:(del_rw+win2)]
                    
for w_ind in range(0,len(w_axe)):
    exp_g=np.exp(-1j*w_axe[w_ind]*gw_ind_mat)
    exp_r=np.exp(-1j*w_axe[w_ind]*rw_ind_mat)

    gwcr_data=gwr_mat*np.real(exp_g)-gwi_mat*np.imag(exp_g)
    gwci_data=gwr_mat*np.imag(exp_g)+gwi_mat*np.real(exp_g)
    rwcr_data=rwr_mat*np.real(exp_r)-rwi_mat*np.imag(exp_r)
    rwci_data=rwr_mat*np.imag(exp_r)+rwi_mat*np.real(exp_r)

    for pulse_counter in range(0,num_pulses):
        if pulse_counter==0:
            [gwcr_data[pulse_counter,:], gwcr_zf]=signal.sosfilt(sos_coefs, gwcr_data[pulse_counter,:], zi=sos_zi)
            [gwci_data[pulse_counter,:], gwci_zf]=signal.sosfilt(sos_coefs, gwci_data[pulse_counter,:], zi=sos_zi)
            [rwcr_data[pulse_counter,:], rwcr_zf]=signal.sosfilt(sos_coefs, rwcr_data[pulse_counter,:], zi=sos_zi)
            [rwci_data[pulse_counter,:], rwci_zf]=signal.sosfilt(sos_coefs, rwci_data[pulse_counter,:], zi=sos_zi)

        else:
            [gwcr_data[pulse_counter,:], gwcr_zf]=signal.sosfilt(sos_coefs, gwcr_data[pulse_counter,:], zi=gwcr_zf)
            [gwci_data[pulse_counter,:], gwci_zf]=signal.sosfilt(sos_coefs, gwci_data[pulse_counter,:], zi=gwci_zf)
            [rwcr_data[pulse_counter,:], rwcr_zf]=signal.sosfilt(sos_coefs, rwcr_data[pulse_counter,:], zi=rwcr_zf)
            [rwci_data[pulse_counter,:], rwci_zf]=signal.sosfilt(sos_coefs, rwci_data[pulse_counter,:], zi=rwci_zf)

        temp1=gwcr_data[pulse_counter,:]
        temp1=temp1[::-1]
        gwcr_data[pulse_counter,:]=temp1
        temp2=gwci_data[pulse_counter,:]
        temp2=temp2[::-1]
        gwci_data[pulse_counter,:]=temp2
        temp3=rwcr_data[pulse_counter,:]
        temp3=temp3[::-1]
        rwcr_data[pulse_counter,:]=temp3
        temp4=rwci_data[pulse_counter,:]
        temp4=temp4[::-1]
        rwci_data[pulse_counter,:]=temp4

        [gwcr_data[pulse_counter,:], gwcr_zf]=signal.sosfilt(sos_coefs, gwcr_data[pulse_counter,:], zi=gwcr_zf)
        [gwci_data[pulse_counter,:], gwci_zf]=signal.sosfilt(sos_coefs, gwci_data[pulse_counter,:], zi=gwci_zf)
        [rwcr_data[pulse_counter,:], rwcr_zf]=signal.sosfilt(sos_coefs, rwcr_data[pulse_counter,:], zi=rwcr_zf)
        [rwci_data[pulse_counter,:], rwci_zf]=signal.sosfilt(sos_coefs, rwci_data[pulse_counter,:], zi=rwci_zf)

        temp1=gwcr_data[pulse_counter,:]
        temp1=temp1[::-1]
        gwcr_data[pulse_counter,:]=temp1
        temp2=gwci_data[pulse_counter,:]
        temp2=temp2[::-1]
        gwci_data[pulse_counter,:]=temp2
        temp3=rwcr_data[pulse_counter,:]
        temp3=temp3[::-1]
        rwcr_data[pulse_counter,:]=temp3
        temp4=rwci_data[pulse_counter,:]
        temp4=temp4[::-1]
        rwci_data[pulse_counter,:]=temp4

    # the second data parsing block

    gwc_data= gwcr_data + gwci_data*1j
    rwc_data= rwcr_data + rwci_data*1j

    gwp_data=np.angle(gwc_data)
    gwa_data=np.absolute(gwc_data)
    rwp_data=np.angle(rwc_data)
    rwa_data=np.absolute(rwc_data)


    gwp_data=np.unwrap(gwp_data,axis=1)
    rwp_data=np.unwrap(rwp_data,axis=1)

    gw_amp_mat[w_ind,:]=np.mean(gwa_data,axis=1)
    rw_amp_mat[w_ind,:]=np.mean(rwa_data,axis=1)
    gw_pha_mat[w_ind,:]=np.mean(gwp_data,axis=1)
    rw_pha_mat[w_ind,:]=np.mean(rwp_data,axis=1)
    
	#print(str(w_ind+1)+"/"+str(len(w_axe)))
    sys.stdout.write("\r"+str(w_ind+1)+"/"+str(len(w_axe))) # The simpliest...
    sys.stdout.flush()										# progressbar
    

sys.stdout.write("\nComplete!\n")

gw_pha_mat=np.unwrap(gw_pha_mat,axis=1)
rw_pha_mat=np.unwrap(rw_pha_mat,axis=1)

mdict={}
mdict['Phi']=rw_pha_mat
mdict['f_axe']=np.arange(f_min,-f_min+f_step,f_step)+f_center
mdict['t_axe']=np.arange(period/fd,period/fd*(num_pulses+1),period/fd)
io.savemat(mat_filename,mdict)

doppler_shifts=np.diff(rw_pha_mat)/2/constants.pi/(period/fd)

head_z="nx " + str(num_pulses-1) + " ny " + str(len(w_axe)) + " xmin " + str(period/fd) + " xmax " + str(num_pulses*period/fd) + " ymin " + str((f_min+f_center)/1000) + " ymax " + str((f_min+f_center)/1000+f_bandwidth/1000)
np.savetxt(
    z_filename,           # file name
    doppler_shifts,                # array to save
    fmt='%.6f',             # formatting, 2 digits in this case
    delimiter='\t',          # column delimiter
    newline='\n',           # new line character
    comments='! ',          # character to use for comments
    header=head_z)

elapsed = time.time() - t
print(elapsed)
