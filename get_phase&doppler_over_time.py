import sys
import time
import struct
import subprocess
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
#from scipy import io

def get_raw_data_from_file(fname,offset_samples,duration_samples, file_parameters=(125000,16,3)):
    if fname[-3::]=='.wv':
        data_raw=subprocess.check_output(
            ['wvunpack','-q','-r',
             '--skip='+str(offset_samples),
             '--until=+'+str(duration_samples),
             fname,'-'],
            shell=True)
    if fname[-4::]=='.bin':        
        sr=file_parameters[0]
        bd=file_parameters[1]
        nc=file_parameters[2]
        offset_bytes=offset_samples*int(bd/8)*nc*2
        duration_bytes=duration_samples*int(bd/8)*nc*2
#        print(duration_samples,bd,nc,duration_bytes)
        f = open(fname, 'rb')
        fsize=f.seek(0,2)
        f.seek(offset_bytes)        
        data_raw=f.read(duration_bytes)
        f.close()
    if fname[-4::]=='.wav':
        print('oops')
        
    return data_raw

# Input parameters
t = time.time()
wv_filename="04061417_part01.wv"
bin_filename="04061417.bin"
fig_filename="doppler_shifts_mpl2.png"
file_parameters=(1000000,16,2)
data_filename=bin_filename
z_filename= "doppler_shifts.z"
mat_filename="phase.mat"
n_ch=file_parameters[2]*2 # number of channels
fd=file_parameters[0] # sample frequency (Hz)
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
#scale_factor=0.5/32768
scale_factor=0.5/32767
win=4000 # window size in samples
win2=1000 # window2 size in samples
del_rw=2550 # delay of the reflected signal in samples
vmin=-0.2
vmax=0.9

# Arrray initializations
w_axe1=np.arange(f_min+fd,fd,f_step)*2*np.pi/fd
w_axe2=np.arange(0,-f_min+f_step,f_step)*2*np.pi/fd
w_axe=np.concatenate((w_axe1,w_axe2), axis=0)

#gwr_mat=np.zeros((num_pulses, win2))
#gwi_mat=np.zeros((num_pulses, win2))
rwr_mat=np.zeros((num_pulses, win2))
rwi_mat=np.zeros((num_pulses, win2))
#gw_ind_mat=np.zeros((num_pulses, win2))
rw_ind_mat=np.zeros((num_pulses, win2))

#gw_amp_mat=np.zeros((len(w_axe),num_pulses))
#rw_amp_mat=np.zeros((len(w_axe),num_pulses))
#gw_pha_mat=np.zeros((len(w_axe),num_pulses))
rw_pha_mat=np.zeros((len(w_axe),num_pulses))
doppler_shifts=np.zeros((len(w_axe),num_pulses-1))

#until_par="--until=+" + str(win)
unpack_str="<"+str(n_ch*win)+"h"

#gwcr_zf=np.zeros((2,2))
#gwci_zf=np.zeros((2,2))
#rwcr_zf=np.zeros((2,2))
#rwci_zf=np.zeros((2,2))

rwcr_zf_list=[np.zeros((2,2)) for i in range(len(w_axe))]
rwci_zf_list=[np.zeros((2,2)) for i in range(len(w_axe))]


elapsed1 = time.time() - t
print("End Preparing State")
print(elapsed1)

# Filling of Arrays (DATA loading)
for pulse_counter in range(0,num_pulses):
    skip_val=int(offset_bytes/2/n_ch)+int(period*(pulse_counter))
    #ind_axe_start=skip_val+strange_shift
    ind_axe_start=skip_val
    ind_axe_all=np.arange(ind_axe_start,ind_axe_start+win) # samples axe for 4000-window
    #skip_str="--skip="+str(skip_val)
    #data_bytes=check_output(["wvunpack", "-r", "-q", skip_str, until_par, wv_filename,"-o", "-"])
    data_bytes=get_raw_data_from_file(data_filename, skip_val, win, file_parameters=file_parameters)
    data=struct.unpack(unpack_str,data_bytes)
    data_scaled=np.reshape(data,[int(len(data)/n_ch), n_ch])*scale_factor

    #gwOr_data=data_scaled[0:win2,0] # O-mode ground wave's real data
    #gwOi_data=data_scaled[0:win2,1] # O-mode ground wave's imagenary data
    #gwXr_data=data_scaled[0:win2,2] # X-mode ground wave's real data
    #gwXi_data=data_scaled[0:win2,3] # X-mode ground wave's imagenary
    
    rwOr_data=data_scaled[del_rw:del_rw+win2,0] # O-mode reflected wave's real data
    rwOi_data=data_scaled[del_rw:del_rw+win2,1] # O-mode reflected wave's imagenary data        
    rwXr_data=data_scaled[del_rw:del_rw+win2,2] # X-mode reflected wave's real data
    rwXi_data=data_scaled[del_rw:del_rw+win2,3] # X-mode reflected wave's imagenary data        
    
    #gwr_mat[pulse_counter,:]=gwOr_data+gwXi_data
    #gwi_mat[pulse_counter,:]=gwOi_data-gwXr_data
    
    rwr_mat[pulse_counter,:]=rwOr_data+rwXi_data
    rwi_mat[pulse_counter,:]=rwOi_data-rwXr_data
    
    #gw_ind_mat[pulse_counter,:]=ind_axe_all[0:win2]

    rw_ind_mat[pulse_counter,:]=ind_axe_all[del_rw:(del_rw+win2)]
elapsed2 = time.time()-elapsed1-t
print("End Filling (DATA loading) State")
print(elapsed2)

# Figure preparing
x_axe=np.arange(0.0,doppler_shifts.shape[1]*period/fd,0.1)
y_axe=np.arange((f_center+f_min)/1000,(f_center+f_min)/1000+len(w_axe),f_step/1000)
fig=plt.figure(figsize=(12,8))
ax=plt.axes()
im=plt.pcolormesh(x_axe, y_axe, doppler_shifts,vmin=-0.2, vmax=0.9)
plt.colorbar()
ax.set_xlim(x_axe[0],x_axe[-1])
ax.set_ylim(y_axe[0],y_axe[-1])
ax.set_xticks([0, 30,60,90,120,150,180,210,240,270,300,330])
plt.savefig(fig_filename)

# Processing Block
print("Data processing (loop over frequencies):")
sos_coefs = signal.iirfilter(f_order, (2**.5)*2*l_c_freq/fd, btype='lowpass', analog=False, ftype='bessel', output='sos')

for pulse_counter in range(0,num_pulses):
    #EXP_C=np.exp(-1j*w_axe[:,np.newaxis]*rw_ind_mat[pulse_counter,:])
    #EXP_R=np.real(EXP_C)
    #EXP_I=np.imag(EXP_C)

    Exp_arg=w_axe[:,np.newaxis]*rw_ind_mat[pulse_counter,:]
    EXP_R=np.cos(Exp_arg)
    EXP_I=-np.sin(Exp_arg)

    RWR_MAT=rwr_mat[pulse_counter,:][np.newaxis,:]
    RWI_MAT=rwi_mat[pulse_counter,:][np.newaxis,:]
    
    RWCR_DATA=RWR_MAT*EXP_R-RWI_MAT*EXP_I
    RWCI_DATA=RWR_MAT*EXP_I+RWI_MAT*EXP_R
        
    for w_ind in range(0,len(w_axe)):
        #exp_g=np.exp(-1j*w_axe[w_ind]*gw_ind_mat[pulse_counter,:])
        #exp_r=np.exp(-1j*w_axe[w_ind]*rw_ind_mat[pulse_counter,:])
    
        #gwcr_data=gwr_mat[pulse_counter,:]*np.real(exp_g)-gwi_mat[pulse_counter,:]*np.imag(exp_g)
        #gwci_data=gwr_mat[pulse_counter,:]*np.imag(exp_g)+gwi_mat[pulse_counter,:]*np.real(exp_g)

        #rwcr_data=rwr_mat[pulse_counter,:]*np.real(exp_r)-rwi_mat[pulse_counter,:]*np.imag(exp_r)
        #rwci_data=rwr_mat[pulse_counter,:]*np.imag(exp_r)+rwi_mat[pulse_counter,:]*np.real(exp_r)  

        #[gwcr_data[pulse_counter,:], gwcr_zf_list[w_ind]]=signal.sosfilt(sos_coefs, gwcr_data[pulse_counter,:], zi=gwcr_zf_list[w_ind])
        #[gwci_data[pulse_counter,:], gwci_zf_list[w_ind]]=signal.sosfilt(sos_coefs, gwci_data[pulse_counter,:], zi=gwci_zf_list[w_ind])

        #[rwcr_data, rwcr_zf_list[w_ind]]=signal.sosfilt(sos_coefs, rwcr_data, zi=rwcr_zf_list[w_ind])
        #[rwci_data, rwci_zf_list[w_ind]]=signal.sosfilt(sos_coefs, rwci_data, zi=rwci_zf_list[w_ind])

        [RWCR_DATA[w_ind,:], rwcr_zf_list[w_ind]]=signal.sosfilt(sos_coefs, RWCR_DATA[w_ind,:], zi=rwcr_zf_list[w_ind])
        [RWCI_DATA[w_ind,:], rwci_zf_list[w_ind]]=signal.sosfilt(sos_coefs, RWCI_DATA[w_ind,:], zi=rwci_zf_list[w_ind])
        
        #[gwcr_data[pulse_counter,:], gwcr_zf_list[w_ind]]=signal.sosfilt(sos_coefs, gwcr_data[pulse_counter,:][::-1], zi=gwcr_zf_list[w_ind])
        #[gwci_data[pulse_counter,:], gwci_zf_list[w_ind]]=signal.sosfilt(sos_coefs, gwci_data[pulse_counter,:][::-1], zi=gwci_zf_list[w_ind])

        #[rwcr_data, rwcr_zf_list[w_ind]]=signal.sosfilt(sos_coefs, rwcr_data[::-1], zi=rwcr_zf_list[w_ind])
        #[rwci_data, rwci_zf_list[w_ind]]=signal.sosfilt(sos_coefs, rwci_data[::-1], zi=rwci_zf_list[w_ind])

        [RWCR_DATA[w_ind,:], rwcr_zf_list[w_ind]]=signal.sosfilt(sos_coefs, RWCR_DATA[w_ind,:][::-1], zi=rwcr_zf_list[w_ind])
        [RWCI_DATA[w_ind,:], rwci_zf_list[w_ind]]=signal.sosfilt(sos_coefs, RWCI_DATA[w_ind,:][::-1], zi=rwci_zf_list[w_ind])
        
        # the second data parsing block

        #gwc_data[pulse_counter,:]= gwcr_data[pulse_counter,:] + gwci_data[pulse_counter,:]*1j
        #rwc_data= rwcr_data + rwci_data*1j

        #gwp_data[pulse_counter,:]=np.angle(gwc_data[pulse_counter,:])
        #gwa_data[pulse_counter,:]=np.absolute(gwc_data[pulse_counter,:])

        #rwp_data=np.angle(rwc_data)
        #rwa_data[pulse_counter,:]=np.absolute(rwc_data[pulse_counter,:])


        #gwp_data[pulse_counter,:]=np.unwrap(gwp_data[pulse_counter,:])
        #rwp_data=np.unwrap(rwp_data)

        #gw_amp_mat[w_ind,pulse_counter]=np.mean(gwa_data)
        #gw_pha_mat[w_ind,pulse_counter]=np.mean(gwp_data)

        #rw_amp_mat[w_ind,pulse_counter]=np.mean(rwa_data)   
        #rw_pha_mat[w_ind,pulse_counter]=np.mean(rwp_data)

        #if pulse_counter>0:
            #doppler_shifts[w_ind,pulse_counter-1]=(rw_pha_mat[w_ind,pulse_counter]-rw_pha_mat[w_ind,pulse_counter-1])/2/np.pi/(period/fd)

    RWP_DATA=np.arctan2(RWCI_DATA,RWCR_DATA)
    RWP_DATA=np.unwrap(RWP_DATA)    
    rw_pha_mat[:,pulse_counter]=np.mean(RWP_DATA,axis=1)
    if pulse_counter>0:
            doppler_shifts[:,pulse_counter-1]=(rw_pha_mat[:,pulse_counter]-rw_pha_mat[:,pulse_counter-1])/2/np.pi/(period/fd)        

    #if pulse_counter%10==0 or pulse_counter==num_pulses-1:
    #if pulse_counter==num_pulses-1:
        #im.remove()
        #im=plt.pcolormesh(x_axe, y_axe, doppler_shifts,vmin=vmin, vmax=vmax)
        #plt.savefig(fig_filename)
    #print(str(w_ind+1)+"/"+str(len(w_axe)))
    sys.stdout.write("\r"+str(pulse_counter+1)+"/"+str(num_pulses)) # The simpliest...
    sys.stdout.flush()                                      # progressbar
#gw_pha_mat=np.unwrap(gw_pha_mat,axis=1)
#rw_pha_mat=np.unwrap(rw_pha_mat,axis=1)
sys.stdout.write("\nComplete!\n")
elapsed3 = time.time()-elapsed2-t
print("End Processing State")
print(elapsed3)

# Saving result
#mdict={}
#mdict['Phi']=rw_pha_mat
#mdict['f_axe']=np.arange(f_min,-f_min+f_step,f_step)+f_center
#mdict['t_axe']=np.arange(period/fd,period/fd*(num_pulses+1),period/fd)
#io.savemat(mat_filename,mdict)

#doppler_shifts=np.diff(rw_pha_mat)/2/np.pi/(period/fd)

im.remove()
im=plt.pcolormesh(x_axe, y_axe, doppler_shifts,vmin=vmin, vmax=vmax)
plt.savefig(fig_filename)

head_z="nx " + str(num_pulses-1) + " ny " + str(len(w_axe)) + " xmin " + str(period/fd) + " xmax " + str(num_pulses*period/fd) + " ymin " + str((f_min+f_center)/1000) + " ymax " + str((f_min+f_center)/1000+f_bandwidth/1000)
np.savetxt(
    z_filename,           # file name
    doppler_shifts,                # array to save
    fmt='%.6f',             # formatting, 2 digits in this case
    delimiter='\t',          # column delimiter
    newline='\n',           # new line character
    comments='! ',          # character to use for comments
    header=head_z)
elapsed4 = time.time()-elapsed3-t
print("End Saving State")
print(elapsed4)


elapsed = time.time() - t
print("Total time")
print(elapsed)
