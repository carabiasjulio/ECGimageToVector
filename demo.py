from ecgImage import ECG_image_values
from ecgSignal import ECG_signal2QRS
import glob


# MAIN FUNCTION
if __name__ == "__main__":
    in_path ='/home/ecgProject/input/'
    #in_path = './ECG-Elda/'

    print('Processing ecgtovector')
    types = (in_path + '*.jpg', in_path + '*.pdf')  # the tuple of file types
    files_grabbed = []
    for files in types:
        files_grabbed.extend(glob.glob(files))

    for file in files_grabbed:
        try:
            print('Procesing File: ' + file)
            ecg, Ts = ECG_image_values(file, draw=False)
            print('Ts estimado: ' + str(Ts))
            value = ECG_signal2QRS(ecg, Ts, draw=False)
            print('QTC estimado: ' + str(value))
            file = open(file + '_QTC.txt', "w")
            file.write(str(value))
            file.close()
        except:
            print('Procesing File: ' + file + ' ha Petado')
