# DataFrame packages
import numpy as np
import pandas as pd

# visualization packages
import seaborn as sns
import matplotlib.pyplot as plt

# heart packages
import heartpy
import wfdb
import neurokit2 as nk

# ML packages
from sklearn.ensemble import ExtraTreesClassifier
# from imblearn.over_sampling import RandomOverSampler
 
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, recall_score, precision_score

from sklearn.model_selection import train_test_split

# others
import os
import joblib



def Distances(X1, Y1, X2, Y2):
    distances_results = []
    
    for x1, y1, x2, y2 in zip(X1.values.flatten(), Y1, X2.values.flatten(), Y2):
        result = np.math.sqrt((x2 - x1)**2 + (y2-y1)**2)
        distances_results.append(result)
        
    return distances_results




def Slope(X1, Y1, X2, Y2):
    slope_results = []
    
    for x1, y1, x2, y2 in zip(X1.values.flatten(), Y1, X2.values.flatten(), Y2):

        result = (y2 - y1) / (x2 - x1)
        slope_results.append(result)
        
    return slope_results




def Amplitudes(peak1, peak2):
    amplitudes = np.abs(peak1 - peak2)
    return amplitudes




def intervals(Peaks1, Peaks2):
    
    res = np.abs(Peaks2 - Peaks1)
    return res




def get_ECG_features(peaks1, peaks2):
    
    X1 = peaks2
    Y1 = signals.iloc[peaks2.values.flatten(), 1]

    X2 = peaks1
    Y2 = signals.iloc[peaks1.values.flatten(), 1]
    
    # Calculate Distances
    distances = Distances(X1, Y1, X2, Y2)
    
    # Calculate Slope
    slopes = Slope(X1, Y1, X2, Y2)

    return distances, slopes




def remove_nulls(peaks, rpeaks):
    
    if len(peaks) > len(rpeaks):
        TF_selection = pd.DataFrame(peaks[:len(rpeaks)]).notna().values.flatten()
        new_peaks = pd.DataFrame(peaks[:len(rpeaks)]).dropna()
        new_rpeaks = pd.DataFrame(rpeaks[TF_selection])
        
        return new_peaks, new_rpeaks
    
    elif len(peaks) < len(rpeaks):
        TF_selection = pd.DataFrame(peaks).notna().values.flatten()
        new_peaks = pd.DataFrame(peaks).dropna()
        rpeaks = rpeaks[:len(TF_selection)]
        new_rpeaks = pd.DataFrame(rpeaks[TF_selection])
        
        return new_peaks, new_rpeaks
    
    else:
        total_df = pd.DataFrame(columns=['ECG_R_Peaks'])
        
        total_df['ECG_R_Peaks'] = rpeaks
        total_df['ECG_Peaks'] = peaks
        
        total_df.dropna(inplace=True)
        
        return total_df['ECG_Peaks'], total_df['ECG_R_Peaks']
    
    


PR_distances = []
PR_slopes = []
PR_amplitudes = []

PQ_distances = []
PQ_slopes = []
PQ_amplitudes = []

QS_distances = []
QS_slopes = []
QS_amplitudes = []

RT_distances = []
RT_slopes = []
RT_amplitudes = []

ST_distances = []
ST_slopes = []
ST_amplitudes = []

PS_amplitudes = []
PT_amplitudes = []
TQ_amplitudes = []
RQ_amplitudes = []
RS_amplitudes = []

QR_interval_list = []
RS_interval_list = []
PQ_interval_list = []
QS_interval_list = []
PS_interval_list = []
PR_interval_list = []
ST_interval_list = []
QT_interval_list = []
RT_interval_list = []
PT_interval_list = []

PR_Amp_list = []
PQ_Amp_list = []
RT_Amp_list = []
PS_Amp_list = []
PT_Amp_list = []
QS_Amp_list = []
TQ_Amp_list = []
TS_Amp_list = []
QR_Amp_list = []
RS_Amp_list = []

label = []


path = 'C:\\Users\\Steven20367691\\Desktop\\Team ECG'
# path = 'D:\\University\\Graduation Project Materials and Models\\ecg-id-database-1.0.0'

team = os.listdir(path)
for member in team:
    files = os.listdir(path+'\\'+member)
    print('--->>>', member)
    for file in files:
        if file.split('.')[1] == 'csv':
            print(file)

            
            # try:
            df = pd.read_csv(path+'\\'+member+'\\'+file)
            
            person_name = df.columns[1]
            sample_rate = int(df.iloc[7, 1].split('.')[0])
            df.drop(person_name, inplace=True, axis=1)
            
            df.drop(range(0, 10), inplace=True)
            df['signals'] = df['Name']
            df.drop('Name', inplace=True, axis=1)
            df['signals'] = df['signals'].astype('float')
            df['signals'].dropna(inplace=True)
            
            signals, info = nk.ecg_process(df['signals'], sampling_rate=sample_rate)

            signals, info = nk.ecg_process(signals.iloc[2000:, 1], sampling_rate=sample_rate)
            
            # rpeaks = nk.ecg_findpeaks(signals.iloc[:, 1], sampling_rate=sample_rate)
            
            filtered_data1 = heartpy.filtering.filter_signal(signals.iloc[:, 1], filtertype='bandpass', cutoff=[2.5, 40], sample_rate=sample_rate, order=3)
            corrected_data1 = heartpy.hampel_correcter(filtered_data1, sample_rate=sample_rate)
            final_signal1 = np.array(filtered_data1) + np.array(corrected_data1)
            
            filtered_data2 = heartpy.filtering.filter_signal(final_signal1, filtertype='bandpass', cutoff=[3, 20], sample_rate=sample_rate, order=3)
            corrected_data2 = heartpy.filtering.hampel_correcter(filtered_data2, sample_rate=sample_rate)
            final_signal2 = np.array(filtered_data2) + np.array(corrected_data2)
            
            _, features = nk.ecg_delineate(final_signal2, sampling_rate=sample_rate, method='peak')
            
            rr_peaks = nk.ecg_findpeaks(final_signal2, sampling_rate=sample_rate)
            
            p_peaks, pr_peaks = remove_nulls(features['ECG_P_Peaks'], rr_peaks['ECG_R_Peaks'])
            # print(len(p_peaks), len(pr_peaks))
            
            q_peaks, qr_peaks = remove_nulls(features['ECG_Q_Peaks'], rr_peaks['ECG_R_Peaks'])
            # print(len(q_peaks), len(qr_peaks))
            
            s_peaks, sr_peaks = remove_nulls(features['ECG_S_Peaks'], rr_peaks['ECG_R_Peaks'])
            # print(len(s_peaks), len(sr_peaks))
            
            t_peaks, tr_peaks = remove_nulls(features['ECG_T_Peaks'], rr_peaks['ECG_R_Peaks'])
            # print(len(t_peaks), len(tr_peaks))
            
            
            PR_distances.extend(get_ECG_features(pr_peaks, p_peaks)[0])
            PR_slopes.extend(get_ECG_features(pr_peaks, p_peaks)[1])
            PR_amplitudes.extend(Amplitudes(pr_peaks.values.flatten(), p_peaks.values.flatten()))

            PQ_distances.extend(get_ECG_features(p_peaks, q_peaks)[0])
            PQ_slopes.extend(get_ECG_features(p_peaks, q_peaks)[1])
            PQ_amplitudes.extend(Amplitudes(np.array(features['ECG_P_Peaks']), np.array(features['ECG_Q_Peaks'])))

            QS_distances.extend(get_ECG_features(q_peaks, s_peaks)[0])
            QS_slopes.extend(get_ECG_features(q_peaks, s_peaks)[1])
            QS_amplitudes.extend(Amplitudes(np.array(features['ECG_Q_Peaks']), np.array(features['ECG_S_Peaks'])))

            RT_distances.extend(get_ECG_features(tr_peaks, t_peaks)[0])
            RT_slopes.extend(get_ECG_features(tr_peaks, t_peaks)[1])
            RT_amplitudes.extend(Amplitudes(tr_peaks.values.flatten(), t_peaks.values.flatten()))

            ST_distances.extend(get_ECG_features(s_peaks, t_peaks)[0])
            ST_slopes.extend(get_ECG_features(s_peaks, t_peaks)[1])
            ST_amplitudes.extend(Amplitudes(np.array(features['ECG_S_Peaks']), np.array(features['ECG_T_Peaks'])))
        
            PS_amplitudes.extend(Amplitudes(np.array(features['ECG_P_Peaks']), np.array(features['ECG_S_Peaks'])))
            PT_amplitudes.extend(Amplitudes(np.array(features['ECG_T_Peaks']), np.array(features['ECG_P_Peaks'])))
            TQ_amplitudes.extend(Amplitudes(np.array(features['ECG_T_Peaks']), np.array(features['ECG_Q_Peaks'])))
            RQ_amplitudes.extend(Amplitudes(q_peaks.values.flatten(), qr_peaks.values.flatten()))
            RS_amplitudes.extend(Amplitudes(sr_peaks.values.flatten(), s_peaks.values.flatten()))
        
            IQR = intervals(q_peaks.values.flatten(), qr_peaks.values.flatten())
            IRS = intervals(sr_peaks.values.flatten(), s_peaks.values.flatten())
            IPQ = intervals(np.array(features['ECG_P_Peaks']), np.array(features['ECG_Q_Peaks']))
            IQS = intervals(np.array(features['ECG_Q_Peaks']), np.array(features['ECG_S_Peaks']))
            IPS = intervals(np.array(features['ECG_P_Peaks']), np.array(features['ECG_S_Peaks']))
            IPR = intervals(p_peaks.values.flatten(), pr_peaks.values.flatten())
            IST = intervals(np.array(features['ECG_S_Peaks']), np.array(features['ECG_T_Peaks']))
            IQT = intervals(np.array(features['ECG_Q_Peaks']), np.array(features['ECG_T_Peaks']))
            IRT = intervals(tr_peaks.values.flatten(), t_peaks.values.flatten())
            IPT = intervals(np.array(features['ECG_P_Peaks']), np.array(features['ECG_T_Peaks']))
            
            QR_interval_list.extend(IQR)
            RS_interval_list.extend(IRS)
            PQ_interval_list.extend(IPQ)
            QS_interval_list.extend(IQS)
            PS_interval_list.extend(IPS)
            PR_interval_list.extend(IPR)
            ST_interval_list.extend(IST)
            QT_interval_list.extend(IQT)
            RT_interval_list.extend(IRT)
            PT_interval_list.extend(IPT)

            label.extend([member]*len(IQT))
                    
                
#             # except:
#             #     print('error..')
#             #     pass

Extracted_Features_DF = pd.DataFrame(columns=[
    'PR Distances', 'PR Slope', 'PR Amplitude',
    'PQ Distances', 'PQ Slope', 'PQ Amplitude',
    'QS Distances', 'QS Slope', 'QS Amplitude',
    'ST Distances', 'ST Slope', 'ST Amplitude',
    'RT Distances', 'RT Slope', 'RT Amplitude',

    'PS Amplitude', 'PT Amplitude', 'TQ Amplitude',
    'QR Amplitude', 'RS Amplitude'
])

lengths = [len(PR_distances), len(PR_slopes), len(PR_amplitudes)
           , len(PQ_distances), len(PQ_slopes), len(PQ_amplitudes)
           , len(QS_distances), len(QS_slopes), len(QS_amplitudes)
           , len(ST_distances), len(ST_slopes), len(ST_amplitudes)
           , len(RT_distances), len(RT_slopes), len(RT_amplitudes)
           , len(PS_amplitudes), len(PT_amplitudes), len(TQ_amplitudes)
           , len(RQ_amplitudes), len(RS_amplitudes)
           
           , len(QR_interval_list), len(RS_interval_list), len(PQ_interval_list)
           , len(QS_interval_list), len(PS_interval_list), len(PR_interval_list)
           , len(ST_interval_list), len(QT_interval_list), len(RT_interval_list)
           , len(PT_interval_list)
          ]

minimum = min(lengths) - 1


Extracted_Features_DF['PR Distances'] = PR_distances[:minimum]
Extracted_Features_DF['PR Slope'] = PR_slopes[:minimum]
Extracted_Features_DF['PR Amplitude'] = PR_amplitudes[:minimum]

Extracted_Features_DF['PQ Distances'] = PQ_distances[:minimum]
Extracted_Features_DF['PQ Slope'] = PQ_slopes[:minimum]
Extracted_Features_DF['PQ Amplitude'] = PQ_amplitudes[:minimum]

Extracted_Features_DF['QS Distances'] = QS_distances[:minimum]
Extracted_Features_DF['QS Slope'] = QS_slopes[:minimum]
Extracted_Features_DF['QS Amplitude'] = QS_amplitudes[:minimum]

Extracted_Features_DF['ST Distances'] = ST_distances[:minimum]
Extracted_Features_DF['ST Slope'] = ST_slopes[:minimum]
Extracted_Features_DF['ST Amplitude'] = ST_amplitudes[:minimum]

Extracted_Features_DF['RT Distances'] = RT_distances[:minimum]
Extracted_Features_DF['RT Slope'] = RT_slopes[:minimum]
Extracted_Features_DF['RT Amplitude'] = RT_amplitudes[:minimum]

Extracted_Features_DF['PS Amplitude'] = PS_amplitudes[:minimum]
Extracted_Features_DF['PT Amplitude'] = PT_amplitudes[:minimum]
Extracted_Features_DF['TQ Amplitude'] = TQ_amplitudes[:minimum]
Extracted_Features_DF['QR Amplitude'] = RQ_amplitudes[:minimum]
Extracted_Features_DF['RS Amplitude'] = RS_amplitudes[:minimum]

Extracted_Features_DF['QR Interval'] = RS_interval_list[:minimum]
Extracted_Features_DF['RS Interval'] = RS_interval_list[:minimum]
Extracted_Features_DF['PQ Interval'] = PQ_interval_list[:minimum]
Extracted_Features_DF['QS Interval'] = QS_interval_list[:minimum]
Extracted_Features_DF['PS Interval'] = PS_interval_list[:minimum]
Extracted_Features_DF['PR Interval'] = PR_interval_list[:minimum]
Extracted_Features_DF['ST Interval'] = ST_interval_list[:minimum]
Extracted_Features_DF['QT Interval'] = QT_interval_list[:minimum]
Extracted_Features_DF['RT Interval'] = RT_interval_list[:minimum]
Extracted_Features_DF['PT Interval'] = PT_interval_list[:minimum]

Extracted_Features_DF['Person'] = label[:minimum]


Extracted_Features_DF.to_csv('C:\\Users\\Steven20367691\\Desktop\\ecg.csv')
print(Extracted_Features_DF.head(60))

# split the data

df = Extracted_Features_DF.dropna()

X = df.iloc[:, :-1]
y = df['Person']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# sampler = RandomOverSampler()
# sampled_train_X, sampled_train_y = sampler.fit_resample(X_train, y_train)

ExtraTree = ExtraTreesClassifier(n_estimators=200, criterion='entropy', verbose=2)


ExtraTree.fit(X_train, y_train)

preds = ExtraTree.predict(X_test)

print(preds)

# ExtraTree model
ExtraTree_preds = ExtraTree.predict(X_test)
print('accuracy_score:', accuracy_score(ExtraTree_preds, y_test.values))
print('f1_score:', f1_score(y_test.values, ExtraTree_preds, average='weighted'))
print('recall_score:', recall_score(ExtraTree_preds, y_test.values, average='weighted'))
print('precision_score:', precision_score(ExtraTree_preds, y_test.values, average='weighted'))


model_path = 'C:\\Users\\Steven20367691\\Desktop\\'

# Save the model
joblib.dump(ExtraTree, model_path+'Extra tree test 2.h5')

print(y.value_counts())
