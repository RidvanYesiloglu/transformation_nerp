import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FormatStrFormatter

def plot_change_of_objective(f_zks, obj_name, K, N, runno, to_save=False, save_folder=None):
    x1 = np.arange(1,len(f_zks)+1)
    y1 = np.asarray(f_zks)
    
    fig, ax1 = plt.subplots(1, 1)
    fig.suptitle('Change of the Objective Value w.r.t. Epoch No (Objective {}, (K,N)=({},{}))'.format(obj_name,K,N))
    
    ax1.plot(x1, y1)
    ax1.set_ylabel('Objective Value')
    ax1.set_xlabel('Epoch Number')
    plt.show()
    if to_save:
        plt.savefig(os.path.join(save_folder, 'obj_vs_ep_{}.png'.format(runno)),bbox_inches='tight')
    plt.close('all')
    return plt

def plot_change_of_loss(losses, obj_name, K, N, runno, to_save=False, save_folder=None):
    x1 = np.arange(1,len(losses)+1)
    y1 = np.asarray(losses)
    
    fig, ax1 = plt.subplots(1, 1)
    fig.suptitle('Change of the Loss Value w.r.t. Epoch No (Objective {}, (K,N)=({},{}))'.format(obj_name,K,N))
    
    ax1.plot(x1, y1)
    ax1.set_ylabel('Loss Value')
    ax1.set_xlabel('Epoch Number')
    plt.show()
    if to_save:
        plt.savefig(os.path.join(save_folder, 'loss_vs_ep_{}.png'.format(runno)),bbox_inches='tight')
    plt.close('all')
    return plt

def plot_au(fin_autocorr_np, K, N, save_folder, run_number):
    au = np.square(fin_autocorr_np)
    fig, ax2 = plt.subplots(figsize=(16, 8), dpi=80)
    au_mean_at_e_d = au.mean((0,1))[1:]
    au_min_at_e_d = au.min((0,1))[1:]
    au_max_at_e_d = au.max((0,1))[1:]
    xs = np.arange(1, N, 1)
    plt.plot(xs,au_max_at_e_d,'r.-')
    plt.plot(xs,au_mean_at_e_d,'m.-')
    plt.plot(xs,au_min_at_e_d,'b.-')
    plt.xlabel('Delay', fontsize=18)
    plt.xticks(fontsize=16)
    plt.ylabel('Squared Autocorrelation',fontsize=18)
    plt.yticks(fontsize=16)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    plt.title('Max., Mean, and Min. Squared Autocorrelation Values Across Codes at Each Delay',fontsize=22)
    plt.legend(["Max Value", "Mean Value","Min Value"],fontsize=16,bbox_to_anchor=(1.2, 0.6))
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'run_{}_ac_vs_delay.png'.format(run_number)),bbox_inches='tight')
    plt.close('all')
    
    fig2, ax3 = plt.subplots(figsize=(16, 8), dpi=80)
    au_mean_f_e_c = au[:,:,1:].mean((0,2))
    au_min_f_e_c = au[:,:,1:].min((0,2))
    au_max_f_e_c = au[:,:,1:].max((0,2))
    xs = np.arange(1, K+1, 1)
    plt.plot(xs,au_max_f_e_c,'r.-')
    plt.plot(xs,au_mean_f_e_c,'m.-')
    plt.plot(xs,au_min_f_e_c,'b.-')
    plt.xlabel('Code No', fontsize=18)
    plt.xticks(fontsize=16)
    plt.ylabel('Squared Autocorrelation',fontsize=18)
    plt.yticks(fontsize=16)
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    plt.title('Max., Mean, and Min. Squared Autocorrelation Values Across Delays for Each Code',fontsize=22)
    plt.legend(["Max Value", "Mean Value","Min Value"],fontsize=16,bbox_to_anchor=(1.2, 0.6))
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'run_{}_ac_vs_code.png'.format(run_number)),bbox_inches='tight')
    plt.close('all')
    
def plot_cc(fin_crosscorr_np, K, N, save_folder, run_number):
    fig, ax = plt.subplots(figsize=(16, 8), dpi=80)
    cc = np.square(fin_crosscorr_np)
    cc_mean = cc.mean((0,1))
    cc_min = cc.min((0,1))
    cc_max = cc.max((0,1))
    xs = np.arange(0, N, 1)
    plt.plot(xs,cc_max,'r.-')
    plt.plot(xs,cc_mean,'m.-')
    plt.plot(xs,cc_min,'b.-')
    plt.xlabel('Delay', fontsize=18)
    plt.xticks(fontsize=16)
    plt.ylabel('Squared Cross-correlation',fontsize=18)
    plt.yticks(fontsize=16)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    plt.title('Max., Mean, and Min. Squared Cross-correlation Values Across Codes at Each Delay',fontsize=22)
    plt.legend(["Max Value", "Mean Value","Min Value"],fontsize=16,bbox_to_anchor=(1.2, 0.6))
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'run_{}_cc_vs_delay.png'.format(run_number)),bbox_inches='tight')  
    plt.close('all')
    
    fig2, ax2 = plt.subplots(figsize=(16, 8), dpi=80)
    cc_mean_f_e_c = cc.mean((0,2))
    cc_min_f_e_c = cc.min((0,2))
    cc_max_f_e_c = cc.max((0,2))
    xs = np.arange(1, K*(K-1)//2 + 1, 1)
    plt.plot(xs,cc_max_f_e_c,'r.-')
    plt.plot(xs,cc_mean_f_e_c,'m.-')
    plt.plot(xs,cc_min_f_e_c,'b.-')
    plt.xlabel('No of 2 Codes', fontsize=18)
    plt.xticks(fontsize=16)
    plt.ylabel('Squared Cross-correlation',fontsize=18)
    plt.yticks(fontsize=16)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    plt.title('Max., Mean, and Min. Squared Cross-correlation Values Across Delays for Each 2 Codes',fontsize=22)
    plt.legend(["Max Value", "Mean Value","Min Value"],fontsize=16,bbox_to_anchor=(1.2, 0.6))
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'run_{}_cc_vs_code.png'.format(run_number)),bbox_inches='tight')
    plt.close('all')