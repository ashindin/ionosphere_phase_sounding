import sys
import time
import struct
import subprocess
import multiprocessing
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

import PyQt5
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.ptime as ptime
import cubehelix
import jet
import viridis

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


def worker_process(ind,e1,e_w,
sos_coefs, win2, num_pulses, period, fd,
rwr_mat,rwi_mat,rw_ind_mat,
w_axe,rwcr_zf_list,rwci_zf_list,
results):
    
    rw_pha_mat=np.zeros((len(w_axe),num_pulses))
    doppler_shifts=np.zeros((len(w_axe),num_pulses-1))
    
    Exp_R_arg=np.zeros((len(w_axe),win2))
    EXP_R_r=np.zeros((len(w_axe),win2))
    EXP_R_i=np.zeros((len(w_axe),win2))
    RWR_MAT=np.zeros((len(w_axe),win2))
    RWI_MAT=np.zeros((len(w_axe),win2))
    RWCR_DATA=np.zeros((len(w_axe),win2))
    RWCI_DATA=np.zeros((len(w_axe),win2))    
    RWP_DATA=np.zeros((len(w_axe),win2))
    RWP_DATA= np.zeros((len(w_axe),win2))
    
    for pulse_counter in range(0,num_pulses):
        e1.wait() # ждем разрешения обрабатывать следующий импульс
        e_w[ind].clear()

        Exp_R_arg=w_axe[:,np.newaxis]*rw_ind_mat[pulse_counter,:]
        EXP_R_r=np.cos(Exp_R_arg)
        EXP_R_i=-np.sin(Exp_R_arg)
            
        RWR_MAT=rwr_mat[pulse_counter,:][np.newaxis,:]
        RWI_MAT=rwi_mat[pulse_counter,:][np.newaxis,:]    
        RWCR_DATA=RWR_MAT*EXP_R_r-RWI_MAT*EXP_R_i
        RWCI_DATA=RWR_MAT*EXP_R_i+RWI_MAT*EXP_R_r    
            
        for w_ind in range(0,len(w_axe)):
    
            [RWCR_DATA[w_ind,:], rwcr_zf_list[w_ind]]=signal.sosfilt(sos_coefs, RWCR_DATA[w_ind,:], zi=rwcr_zf_list[w_ind])
            [RWCI_DATA[w_ind,:], rwci_zf_list[w_ind]]=signal.sosfilt(sos_coefs, RWCI_DATA[w_ind,:], zi=rwci_zf_list[w_ind])
     
        RWP_DATA=np.arctan2(RWCI_DATA,RWCR_DATA)
        RWP_DATA=np.unwrap(RWP_DATA)    
        #rw_pha_mat[:,pulse_counter]=np.mean(RWP_DATA,axis=1)
        rw_pha_mat[:,pulse_counter]=RWP_DATA[:,499]

        if pulse_counter>0:
                doppler_shifts[:,pulse_counter-1]=(rw_pha_mat[:,pulse_counter]-rw_pha_mat[:,pulse_counter-1])/2/np.pi/(period/fd)        
                results.put((ind, doppler_shifts[:,pulse_counter-1])) # отправляем результат в очередь
        e_w[ind].set() # сообщаем в main, что импульс обработан        

    

if __name__ == '__main__': 
       
    # Input parameters
    t = time.time()
    wv_filename="04061417_part01.wv"
    bin_filename="04061417.bin"
    fig_filename="doppler_shifts_mpl_mp2.png"
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
    w_axe2=np.arange(0,-f_min,f_step)*2*np.pi/fd
    w_axe=np.concatenate((w_axe1,w_axe2), axis=0)
    
    rwr_mat=np.zeros((num_pulses, win2))
    rwi_mat=np.zeros((num_pulses, win2))

    rw_ind_mat=np.zeros((num_pulses, win2))

    #rw_pha_mat=np.zeros((len(w_axe),num_pulses))
    doppler_shifts=np.zeros((len(w_axe),num_pulses-1))
    
    unpack_str="<"+str(n_ch*win)+"h"
    
    rwcr_zf_list=[np.zeros((2,2)) for i in range(len(w_axe))]
    rwci_zf_list=[np.zeros((2,2)) for i in range(len(w_axe))]

    
    
    elapsed1 = time.time() - t
    print("End Preparing State")
    print(elapsed1)
    
    # Filling of Arrays (DATA loading)
    for pulse_counter in range(0,num_pulses):
        skip_val=int(offset_bytes/2/n_ch)+int(period*(pulse_counter))
        ind_axe_start=skip_val
        ind_axe_all=np.arange(ind_axe_start,ind_axe_start+win) # samples axe for 4000-window
        data_bytes=get_raw_data_from_file(data_filename, skip_val, win, file_parameters=file_parameters)
        data=struct.unpack(unpack_str,data_bytes)
        data_scaled=np.reshape(data,[int(len(data)/n_ch), n_ch])*scale_factor
    
        
        rwOr_data=data_scaled[del_rw:del_rw+win2,0] # O-mode reflected wave's real data
        rwOi_data=data_scaled[del_rw:del_rw+win2,1] # O-mode reflected wave's imagenary data        
        rwXr_data=data_scaled[del_rw:del_rw+win2,2] # X-mode reflected wave's real data
        rwXi_data=data_scaled[del_rw:del_rw+win2,3] # X-mode reflected wave's imagenary data        
        
        rwr_mat[pulse_counter,:]=rwOr_data+rwXi_data
        rwi_mat[pulse_counter,:]=rwOi_data-rwXr_data

        rw_ind_mat[pulse_counter,:]=ind_axe_all[del_rw:(del_rw+win2)]
    elapsed2 = time.time()-elapsed1-t
    print("End Filling (DATA loading) State")
    print(elapsed2)
    
    sos_coefs = signal.iirfilter(f_order, (2**.5)*2*l_c_freq/fd, btype='lowpass', analog=False, ftype='bessel', output='sos')
    # MULTIPROCESSING
    
    n_cpu=multiprocessing.cpu_count()    
    chunks=[round(len(w_axe)/n_cpu) for i in range(n_cpu-1)]
    chunks.append(len(w_axe)-sum(chunks))
    
    ch_id=[]
    for i in range(len(chunks)):
        if i==0:
            st_ind=0
        else:
            st_ind=sum(chunks[0:i])
        en_ind=st_ind+chunks[i]
        ch_id.append(( st_ind,en_ind))
    print(ch_id)
    
    results = multiprocessing.Queue()
    e1 = multiprocessing.Event()
    e_w = []
    w=[]
    for i in range(n_cpu):
        e_w.append(multiprocessing.Event())
        w.append( multiprocessing.Process(name='worker '+str(i), 
                                 target=worker_process,
                                 args=(
                                 i,e1,e_w,
                                 sos_coefs, win2, num_pulses, period, fd,
                                 rwr_mat, rwi_mat, rw_ind_mat, 
                                 w_axe[ch_id[i][0]:ch_id[i][1]],
                                 rwcr_zf_list[ch_id[i][0]:ch_id[i][1]], rwci_zf_list[ch_id[i][0]:ch_id[i][1]],
                                 results
                                 )))
        w[i].start() 
    ### GUI BLOCK    
    app = QtGui.QApplication([])
    ## Create window with GraphicsView widget
    win = pg.GraphicsLayoutWidget()
    win.show()  ## show widget alone in its own window
    win.setWindowTitle('Doppler shifts')
    view = win.addViewBox()
    ## lock the aspect ratio so pixels are always square
    view.setAspectLocked(True)
    ## Create image item
    img = pg.ImageItem(border='w')
    view.addItem(img)
    #img.setLookupTable(lut_cubehelix)
    ## Set initial view bounds
    im_width=660
    im_height=425
    view.setRange(QtCore.QRectF(0, 0, im_width, im_height))
    doppler_shifts_plot=np.zeros((int(num_pulses/5),int(len(w_axe)/2)))
    
    def updateData():
        global img, doppler_shifts_plot    
        #img.setImage(doppler_shifts_plot,autoLevels=False, levels=[-0.2, 0.9], lut=viridis.viridis())
        img.setImage(doppler_shifts_plot,autoLevels=False, levels=[-0.2, 0.9], lut=jet.jet())
        QtCore.QTimer.singleShot(1, updateData)


    QtGui.QApplication.processEvents()
    
    # Processing Block
    print("Data processing (loop over pulses/time):")    
    
    #print([e.is_set() for e in e_w])
    for pulse_counter in range(0,num_pulses):
        #print(pulse_counter)
        e1.set() # разрешаем вёкерам обрабатывать следующий импульс
        e1.clear() # Запрещаем вёкерам обрабатывать импульс
        while True:
            #time.sleep(1)
            #print(pulse_counter, [e.is_set() for e in e_w])
            if all([e.is_set() for e in e_w]): break # ждем пока ВСЕ вёкеры обработают импульс        
            
        
        # сбор результатов
        #print(pulse_counter, results.qsize())
        if pulse_counter>0:
            num_jobs=n_cpu
            while num_jobs:
                result = results.get()
                i=result[0]
                doppler_shifts[ch_id[i][0]:ch_id[i][1],pulse_counter-1]=result[1]
                num_jobs -= 1            
        if pulse_counter>1:
            if (pulse_counter)%5==0 or pulse_counter==num_pulses-1:
                doppler_shifts_plot[int((pulse_counter)/5)-1,:]=doppler_shifts[0::2,pulse_counter-1].T
                updateData()
                QtGui.QApplication.processEvents()
        
        sys.stdout.write("\r"+str(pulse_counter+1)+"/"+str(num_pulses)) # The simpliest...
        sys.stdout.flush()                                      # progressbar
    
    sys.stdout.write("\nComplete!\n")
    elapsed3 = time.time()-elapsed2-t
    print("End Processing State")
    print(elapsed3)
    
    # Figure
    x_axe=np.arange(0.0,doppler_shifts.shape[1]*period/fd,0.1)
    y_axe=np.arange((f_center+f_min)/1000,(f_center+f_min)/1000+len(w_axe),f_step/1000)
    fig=plt.figure(figsize=(12,8))
    ax=plt.axes()
    im=plt.pcolormesh(x_axe, y_axe, doppler_shifts,vmin=-0.2, vmax=0.9, cmap='jet')
    plt.colorbar()
    ax.set_xlim(x_axe[0],x_axe[-1])
    ax.set_ylim(y_axe[0],y_axe[-1])
    ax.set_xticks([0, 30,60,90,120,150,180,210,240,270,300,330])
    plt.savefig(fig_filename)
    
    # Saving result
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
        
    
