import numpy as np
from matplotlib import pyplot as plt
import glob
import scipy.io as sio
import wfdb
import subprocess
# plt.switch_backend('MacOSX')

# ---------------------------------------------------------------------------------------
# Funciones Auxiliares
# ---------------------------------------------------------------------------------------
def plotEstimation(ecg, tm, Q, T, R):
    plt.plot(tm, ecg, zorder=1)
    plt.scatter([tm[i] for i in R], [ecg[i] for i in R], c='r', marker='*', zorder=2)
    plt.scatter([tm[i] for i in Q], [ecg[i] for i in Q], c='m', marker='x', zorder=2)
    plt.scatter([tm[i] for i in T], [ecg[i] for i in T], c='m', marker='o', zorder=2)
    plt.show()

# ---------------------------------------------------------------------------------------
# Funciones Principales
# ---------------------------------------------------------------------------------------
def readAnnotation(name):
    # LECTURA DEL FICHERO DE ANOTACION
    annotation = wfdb.rdann(name, 'test')

    anntype = annotation.symbol
    ann = annotation.sample

    # LEYENDO R
    R_est_ann = np.char.find(anntype, 'N')
    R_est = [ann[i] for i in range(0, len(R_est_ann)) if R_est_ann[i] == 0]

    # LEYENDO Q
    Q_ini_est_ann = np.char.find(anntype, 'N')
    Q_ini_est = [ann[i - 1] for i in range(0, len(Q_ini_est_ann)) if Q_ini_est_ann[i] == 0 and anntype[i - 1] == '(']

    # LEYENDO T
    T_end_est_ann = np.char.find(anntype, 't')
    T_end_est = [ann[i + 1] for i in range(0, len(T_end_est_ann)) if T_end_est_ann[i] == 0 and anntype[i + 1] == ')']

    R_est_ann = [i for i in range(0, len(R_est_ann)) if R_est_ann[i] == 0] # SE UTILIZA EN EL CALCULO QT

    return Q_ini_est, T_end_est, R_est, anntype, ann, R_est_ann


def computeQTtimes(anntype, ann, R, tm):
    rr_qt_matrix = np.zeros([len(R),4],dtype=int)
    rr_qt_index = -1

    for rinx in range(len(R)):
        r1 = R[rinx]
        #VARIABLES DE CONTROL
        q_init = np.nan
        t_end = np.nan
        r2 = np.nan

        if r1 > 0: # EVITAR VALORES NEGATIVOS
            if anntype[r1-1] == '(': # Q ENCONTRADO
                q_init = r1-1

        ann_add = 1
        if ~np.isnan(q_init):
            while np.isnan(r2) and ((r1+ann_add)<(len(anntype)-1)):
                if anntype[r1+ann_add] =='N': # SIGUIENTE R ENCONTRADO
                    r2 = r1+ann_add
                elif (anntype[r1+ann_add]+anntype[r1+ann_add+1] == 't)') and np.isnan(t_end): # T_END ENCONTRADO
                    t_end = r1+ann_add+1

                ann_add = ann_add + 1

            # GUARDA PARAMETROS
            if ~np.isnan(q_init) and ~np.isnan(t_end) and ~np.isnan(r2):
                rr_qt_index = rr_qt_index+1
                rr_qt_matrix[rr_qt_index, :] = [r1,r2,q_init,t_end]

    # CALCULO DEL QTC
    if rr_qt_index > -1:
        rr_qt_matrix = rr_qt_matrix[:rr_qt_index+1,:]
        rr_qt_times = np.zeros((rr_qt_index+1,4))

        for i in range(rr_qt_index+1):
            for j in range(4):
                rr_qt_times[i, j] = tm[ann[rr_qt_matrix[i,j]]]

        rr_qt_times[:,0] = rr_qt_times[:,1] - rr_qt_times[:,0]
        rr_qt_times[:,1] = rr_qt_times[:,3] - rr_qt_times[:,2]
        rr_qt_times = rr_qt_times[:,:2]

        qtc_times = np.divide(rr_qt_times[:,1],np.power(rr_qt_times[:,0], 1/3))
        qtc = np.mean(qtc_times)
    else:
        qtc = np.nan

    return qtc


def ECG_signal2QRS(ecg, Ts, draw=False):
    if ~np.isnan(Ts): # ES NECESARIO Ts
        tm = np.arange(0, len(ecg) * Ts, Ts)[:len(ecg)]

        # GENERA FILENAME RANDOM
        filename = 'tmp' + str(int(np.mean(ecg)*10))

        # GUARDA LA SENAL ECG EN FORMATO WFDB
        wfdb.wrsamp(filename, fs=1 / Ts, units=['mV'], sig_name=['I'], p_signal=ecg, fmt=['212'])

        # ESTIMA EL COMPLEJO QRS
        try:
            out = subprocess.Popen(['ecgpuwave','-r',filename,'-a','test'],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
            stdout, stderr = out.communicate()

            # LEE LA ESTIMACION DEL COMPLEJO QRS
            Q, T, R, anntype, ann, R_ann = readAnnotation(filename)

            # DIBUJA LAS ESTIMACION REALIZADA POR ECGPUWAVE
            if draw:
                plotEstimation(ecg, tm, Q, T, R)

            # CALCULO DEL TIEMPO QT
            qtc = computeQTtimes(anntype, ann, R_ann, tm)
        except:
            qtc = np.nan
    else:
        qtc = np.nan

    return qtc


# MAIN FUNCTION
if __name__ == "__main__":

    print('Processing ecgtovector')

    for file in glob.glob("../../Desktop/COVID19/data_Raul/ecgc-set1/*.mat"):
        # print('Procesing File: ' + file)

        # LECTURA FICHERO .MAT
        mat = sio.loadmat(file)
        ecg = mat['ecg']
        ecg = np.transpose(ecg)  # FORMATO: MxN (M:LENGTH,N:CHANNELS)

        Ts = mat['Ts'][0,0]

        # FUNCION PRINCIPAL
        value = ECG_signal2QRS(ecg, Ts, draw=False)

        # print('Valor estimado: ' + str(value))

        print(file + ' ' + str(value))