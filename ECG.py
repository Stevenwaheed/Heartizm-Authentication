# API packages
from flask import Flask, jsonify, request, make_response
from flask_restful import Api

# DataFrame packages
import numpy as np
import pandas as pd

# heart packages
import neurokit2 as nk
import heartpy

# ML packages
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split

# Database packages
import sqlite3
import _sqlite3

# others
import joblib
import json
import secrets



'''
    This class contains all necessary functions that needed to deal with ECG signals such as:
    1- Distances function:                                    that calculate the distance between peaks.
    2- Slope function:                                        that calculate the slope of the line between peaks.
    3- Interval function:                                     that calculate the Interval between peaks.
    4- Amplitude function:                                    that calculate the Amplitude between peaks.
    5- get_ECG_features function:                             apply (Distances, Slope, Interval, Amplitude) functions on the ECG signals.
    6- remove_nulls function:                                 remove the nulls from the ECG signals without affect the signal behavior.
    7- get_sample_rate function:                              return the heart rate from the csv file.
    8- get_person_name function:                              return the person from the csv file for identification.
    9- edit_dataframe function:                               edit the csv file to get only the signal values.
    10- peak_detection function:                              this function detect the main peaks (P, QRS, T) after cleaning the singal.
    11- feature_exctraction function:                         get the main features such as (Distances between peaks, Slope, Amplitude, Intervals).
    12- identification_labled_feature_exctraction function:   extract the main features with persons' names for identification.
    13- authentication_labled_feature_exctraction function:   extract the main features for authenticated person and assign it with label 1.
'''
class ECG:
    
    def __init__(self, original_signal):
        self.original_signal = original_signal    #    It is a DateFrame.
        
    
    '''
        this function  takes four parameters point1: (X1, Y1)(an ECG peak), point2: (X2, Y2)(an ECG peak) 
        to get the distances between ecg peaks, and return the {distance} between these two peaks.
        {PR Distances, PQ Distances, QS Distances, ST Distances, RT Distances}.
    '''
    def Distances(self, X1, Y1, X2, Y2):
        distances_results = []
        
        for x1, y1, x2, y2 in zip(X1.values.flatten(), Y1, X2.values.flatten(), Y2):
            result = np.math.sqrt((x2 - x1)**2 + (y2-y1)**2)
            distances_results.append(result)
            
        return distances_results


    '''
        this function takes four parameters point1: (X1, Y1)(an ECG peak), point2: (X2, Y2)(an ECG peak) 
        to get the slope of the lines between ecg peaks, and return the {slope} between these two peaks.
        {PR Slope, PQ Slope, QS Slope, ST Slope, RT Slope}.
    '''
    def Slope(self, X1, Y1, X2, Y2):
        slope_results = []
        
        for x1, y1, x2, y2 in zip(X1.values.flatten(), Y1, X2.values.flatten(), Y2):

            result = (y2 - y1) / (x2 - x1)
            slope_results.append(result)
            
        return slope_results


    '''
        this function takes two ECG peaks and gets the amplitudes between ecg peaks from the dataframe that we've created.
        and return the {total amplitude} between these two waves.
        {PR Amplitude, PQ Amplitude, QS Amplitude, ST Amplitude
        , RT Amplitude, PS Amplitude, PT Amplitude, TQ Amplitude
        ,QR Amplitude, RS Amplitude}.
    '''
    def Amplitudes(self, peak1, peak2):
        amplitudes = np.abs(peak1 - peak2)
        return amplitudes


    '''
        Intervals function to get the output of difference between heart peaks on x axis.
        return {the intervals}
    '''
    def intervals(self, Peaks1, Peaks2):
        res = np.abs(Peaks2 - Peaks1)
        return res


    '''
        This function takes two ECG peaks, and return the {distances and slopes} between peaks.
    '''
    def get_ECG_features(self, peaks1, peaks2):
        signal = self.edit_dataframe()
        signal.dropna(inplace=True)
        
        X1 = peaks2
        # print(peaks2.values.flatten())
        Y1 = signal.iloc[peaks2.values.flatten(), 0]

        X2 = peaks1
        Y2 = signal.iloc[peaks1.values.flatten(), 0]
        
        # Calculate Distances
        distances = self.Distances(X1, Y1, X2, Y2)
        
        # Calculate Slope
        slopes = self.Slope(X1, Y1, X2, Y2)

        return distances, slopes

    '''
        thsi function take the csv file (original_signal){It is a DateFrame} and returns {heart rate}.
    '''
    def get_sample_rate(self):
        sample_rate = int(self.original_signal.iloc[7, 1].split('.')[0])
        return sample_rate


    '''
        thsi function take the csv file (original_signal){It is a DateFrame} and returns {person name}.
    '''
    def get_person_name(self):
        person_name = self.original_signal.columns[1]
        return person_name


    '''
        this function remove the second column (NULLS column) and take the first one(the signal values),
        drop the first ten rows and start from the beginning of the signal.
        
        return {the ECG signal values}.
    '''
    def edit_dataframe(self):
        cols = self.original_signal.columns
        original_signal_1 = self.original_signal.drop(cols[1], axis=1)

        original_signal_2 = original_signal_1.drop(range(0, 10))
        original_signal_2['signals'] = original_signal_2['Name']
        
        original_signal_3 = original_signal_2.drop('Name', axis=1)
        original_signal_3['signals'] = original_signal_3['signals'].astype('float')
        original_signal_3 = original_signal_3.dropna()
        
        return original_signal_3


    '''
        this function takes two ECG peaks with nulls and removes the nulls from the ECG wave,
        and return the {correct two ECG peaks}.
    '''
    def remove_nulls(self, peaks, rpeaks):
        
        if len(peaks) > len(rpeaks):
            TF_selection = pd.DataFrame(peaks[:len(rpeaks)]).notna().values.flatten()
            new_peaks = pd.DataFrame(peaks[:len(rpeaks)]).dropna()
            new_rpeaks = pd.DataFrame(rpeaks[TF_selection])
            
            return new_peaks.astype('int'), new_rpeaks.astype('int')
        
        elif len(peaks) < len(rpeaks):
            TF_selection = pd.DataFrame(peaks).notna().values.flatten()
            new_peaks = pd.DataFrame(peaks).dropna()
            rpeaks = rpeaks[:len(TF_selection)]
            new_rpeaks = pd.DataFrame(rpeaks[TF_selection])
            
            return new_peaks.astype('int'), new_rpeaks.astype('int')
        
        else:
            total_df = pd.DataFrame(columns=['ECG_R_Peaks'])
            
            total_df['ECG_R_Peaks'] = rpeaks
            total_df['ECG_Peaks'] = peaks
            
            total_df.dropna(inplace=True)
            
            return total_df['ECG_Peaks'].astype('int'), total_df['ECG_R_Peaks'].astype('int')
        

    '''
        This function get the dataframe of the signal after the editing it, then perform some operations on it such as:
        1- Get the specific ECG signal after removing the noise from it by determining {the highpass and the lowpass for the ECG signal}.
        2- Normalize using {hampel_correcter}.
        3- Detect the main peaks (P, QRS, T).
        
        and returns the {filtered signal}, the {r peaks} and the {other peaks}.
    '''
    def peak_detection(self):
        sample_rate = self.get_sample_rate()
        signal = self.edit_dataframe()
        signals, _ = nk.ecg_process(signal.values.flatten(), sampling_rate=sample_rate)
        signals, _ = nk.ecg_process(signals['ECG_Clean'][2000:], sampling_rate=sample_rate)
        
        filtered_data = heartpy.filtering.filter_signal(signals.iloc[:, 1], filtertype='bandpass', cutoff=[2.5, 40], sample_rate=sample_rate, order=3)
        corrected_data = heartpy.hampel_correcter(filtered_data, sample_rate=sample_rate)
        final_signal = np.array(filtered_data)+np.array(corrected_data)
       
        filtered_data2= heartpy.filtering.filter_signal(final_signal, filtertype='bandpass', cutoff=[3, 20], sample_rate=sample_rate, order=3)
        corrected_data2 = heartpy.filtering.hampel_correcter(filtered_data2, sample_rate=sample_rate)
        final_signal2 = np.array(filtered_data2) + np.array(corrected_data2)
        
        rr_peaks = nk.ecg_findpeaks(final_signal2, sampling_rate=sample_rate)
        
        _, features = nk.ecg_delineate(final_signal2, sampling_rate=sample_rate, method='peak')
        
        return signals, rr_peaks, features
        

    '''
        This function perform distance, slope, difference between amplitude and intervals formulas  
        to get the main features such as (Distances between peaks, Slope, Amplitude, Intervals).
        
        and returns a {dataframe of all main extracted features for the ECG signal}.
    '''
    def feature_exctraction(self):
        Extracted_Features_DF = pd.DataFrame(columns=[
        'PR Distances', 'PR Slope', 'PR Amplitude',
        'PQ Distances', 'PQ Slope', 'PQ Amplitude',
        'QS Distances', 'QS Slope', 'QS Amplitude',
        'ST Distances', 'ST Slope', 'ST Amplitude',
        'RT Distances', 'RT Slope', 'RT Amplitude',

        'PS Amplitude', 'PT Amplitude', 'TQ Amplitude',
        'QR Amplitude', 'RS Amplitude'
        ])
        
        _, rr_peaks, features = self.peak_detection()
        
        
        p_peaks, pr_peaks = self.remove_nulls(features['ECG_P_Peaks'], rr_peaks['ECG_R_Peaks'])
        q_peaks, qr_peaks = self.remove_nulls(features['ECG_Q_Peaks'], rr_peaks['ECG_R_Peaks'])
        s_peaks, sr_peaks = self.remove_nulls(features['ECG_S_Peaks'], rr_peaks['ECG_R_Peaks'])
        t_peaks, tr_peaks = self.remove_nulls(features['ECG_T_Peaks'], rr_peaks['ECG_R_Peaks'])
 
        
        # Features between PR
        PR_distances = self.get_ECG_features(pr_peaks, p_peaks)[0]
        PR_slopes = self.get_ECG_features(pr_peaks, p_peaks)[1]
        PR_amplitudes = self.Amplitudes(pr_peaks.values.flatten(), p_peaks.values.flatten())

        # Features between PQ
        PQ_distances = self.get_ECG_features(p_peaks, q_peaks)[0]
        PQ_slopes = self.get_ECG_features(p_peaks, q_peaks)[1]
        PQ_amplitudes = self.Amplitudes(np.array(features['ECG_P_Peaks']), np.array(features['ECG_Q_Peaks']))

        # Features between QS
        QS_distances = self.get_ECG_features(q_peaks, s_peaks)[0]
        QS_slopes = self.get_ECG_features(q_peaks, s_peaks)[1]
        QS_amplitudes = self.Amplitudes(np.array(features['ECG_Q_Peaks']), np.array(features['ECG_S_Peaks']))

        # Features between RT
        RT_distances = self.get_ECG_features(tr_peaks, t_peaks)[0]
        RT_slopes = self.get_ECG_features(tr_peaks, t_peaks)[1]
        RT_amplitudes = self.Amplitudes(tr_peaks.values.flatten(), t_peaks.values.flatten())

        # Features between ST
        ST_distances = self.get_ECG_features(s_peaks, t_peaks)[0]
        ST_slopes = self.get_ECG_features(s_peaks, t_peaks)[1]
        ST_amplitudes = self.Amplitudes(np.array(features['ECG_S_Peaks']), np.array(features['ECG_T_Peaks']))
    
        # the other amplitude features 
        PS_amplitudes = self.Amplitudes(np.array(features['ECG_P_Peaks']), np.array(features['ECG_S_Peaks']))
        PT_amplitudes = self.Amplitudes(np.array(features['ECG_T_Peaks']), np.array(features['ECG_P_Peaks']))
        TQ_amplitudes = self.Amplitudes(np.array(features['ECG_T_Peaks']), np.array(features['ECG_Q_Peaks']))
        RQ_amplitudes = self.Amplitudes(q_peaks.values.flatten(), qr_peaks.values.flatten())
        RS_amplitudes = self.Amplitudes(sr_peaks.values.flatten(), s_peaks.values.flatten())
    
        # intervals features
        QR_interval = self.intervals(q_peaks.values.flatten(), qr_peaks.values.flatten())
        RS_interval = self.intervals(sr_peaks.values.flatten(), s_peaks.values.flatten())
        PQ_interval = self.intervals(np.array(features['ECG_P_Peaks']), np.array(features['ECG_Q_Peaks']))
        QS_interval = self.intervals(np.array(features['ECG_Q_Peaks']), np.array(features['ECG_S_Peaks']))
        PS_interval = self.intervals(np.array(features['ECG_P_Peaks']), np.array(features['ECG_S_Peaks']))
        PR_interval = self.intervals(p_peaks.values.flatten(), pr_peaks.values.flatten())
        ST_interval = self.intervals(np.array(features['ECG_S_Peaks']), np.array(features['ECG_T_Peaks']))
        QT_interval = self.intervals(np.array(features['ECG_Q_Peaks']), np.array(features['ECG_T_Peaks']))
        RT_interval = self.intervals(tr_peaks.values.flatten(), t_peaks.values.flatten())
        PT_interval = self.intervals(np.array(features['ECG_P_Peaks']), np.array(features['ECG_T_Peaks']))
        
        
        # list of lengths of all lists.
        lengths = [len(PR_distances), len(PR_slopes), len(PR_amplitudes)
           , len(PQ_distances), len(PQ_slopes), len(PQ_amplitudes)
           , len(QS_distances), len(QS_slopes), len(QS_amplitudes)
           , len(ST_distances), len(ST_slopes), len(ST_amplitudes)
           , len(RT_distances), len(RT_slopes), len(RT_amplitudes)
           , len(PS_amplitudes), len(PT_amplitudes), len(TQ_amplitudes)
           , len(RQ_amplitudes), len(RS_amplitudes)
           
           , len(QR_interval), len(RS_interval), len(PQ_interval)
           , len(QS_interval), len(PS_interval), len(PR_interval)
           , len(ST_interval), len(QT_interval), len(RT_interval)
           , len(PT_interval)
          ]

        # get the minimum length to make all lists have the same length. 
        minimum = min(lengths) - 1


        # Store the lists of features in the dataframe.
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

        Extracted_Features_DF['QR Interval'] = QR_interval[:minimum]
        Extracted_Features_DF['RS Interval'] = RS_interval[:minimum]
        Extracted_Features_DF['PQ Interval'] = PQ_interval[:minimum]
        Extracted_Features_DF['QS Interval'] = QS_interval[:minimum]
        Extracted_Features_DF['PS Interval'] = PS_interval[:minimum]
        Extracted_Features_DF['PR Interval'] = PR_interval[:minimum]
        Extracted_Features_DF['ST Interval'] = ST_interval[:minimum]
        Extracted_Features_DF['QT Interval'] = QT_interval[:minimum]
        Extracted_Features_DF['RT Interval'] = RT_interval[:minimum]
        Extracted_Features_DF['PT Interval'] = PT_interval[:minimum]
       
        return Extracted_Features_DF
    
    
    '''
        this function create a labeled dataframe and the labels are the persons' names for identification. 
        return a {dataframe}
    '''
    def identification_labled_feature_exctraction(self):
        features = self.feature_exctraction()
        label = self.get_person_name()
        merged_df = pd.concat([features, pd.Series([label]*features.shape[0])], axis=1)
        
        return merged_df
    
    
    '''
        this function create a labeled dataframe and the label is 1 for authenticated persons. 
        return a {dataframe}
    '''
    def authentication_labled_feature_exctraction(self):
        features = self.feature_exctraction()
        merged_df = pd.concat([features, pd.Series([1] * features.shape[0])], axis=1)
        
        return merged_df
        




'''
    This class have the functions that deal with our database such as:
    1- select_command function:            this function have selection command that select some columns from the database.
    2- insert_person_command function:     this function have insertion command that insert new person to the database.
    3- insert_model function:              this function have insertion command that insert new model to the database.
    4- insert_features_command function:   this function have insertion command that insert new main features to the database.
    5- fetch function:                     this function execute the selection command and fetch data from the database.
    6- insert function:                    this function execute the insertion command and insert data to the database.
    7- create function:                    this function create the main tables in our database.
'''
class sql_ecg():
    def __init__(self, person_ID=0, person_name='', email='', phone_number=''):
        self.person_ID = person_ID
        self.person_name = person_name
        self.email = email
        self.phone_number = phone_number
        

    '''
        this function takes the columns that we need to be returned and the table, 
        and add them to the selection commend to be executed.
    '''
    def select_command(self, cols, table):
        return 'SELECT '+ cols + ' FROM ' + table
    
    
    '''
        this function add the ID, person name, email and phone number to the insertion commend to be executed.
    '''
    def insert_person_command(self):
        return 'INSERT INTO Person VALUES ("' + str(self.person_ID) + '", "' + self.person_name + '", "' + self.email + '", "' + self.phone_number + '", "Model")'
    
    
    '''
        this function takes the model that we need to be inserted, 
        and insert it to the specific person.
        
        save the model in the same path of the project.
    '''
    def insert_model(self, model):
        model_name = str(self.person_ID) + self.person_name + self.phone_number+'.h5'

        db = sqlite3.connect('Heartizm.db')
        cursor = db.cursor()

        cursor.execute('UPDATE Person SET ML_model="'+ model_name +'" WHERE ID="'+str(self.person_ID)+'"')

        db.commit()
        db.close()

        joblib.dump(model, model_name)

    
    '''
        this function takes the table that we want to insert the 31 features in it, 
        and add it to the insertion commend to be executed. 
        
        return {the insertion command}.
    '''
    def insert_features_command(self, table):
        columns = '?, '*31
        return 'INSERT INTO '+ table +' VALUES ('+ columns[:-2]+')'
    
    
    '''
        this function takes the columns that we need to be returned and the table, 
        and send them to the selection function that select the data from these columns and that table from the database,
        fetch them in a list of tuples.
        
        return a {dataframe of all features}.
    '''
    def fetch(self, cols, table):
        connection = sqlite3.connect('Heartizm.db')
        cursor = connection.cursor()
        
        command = self.select_command(cols, table)        
        cursor.execute(command)
        fetched_data = cursor.fetchall()

        connection.commit()
        connection.close()
        
        df = pd.DataFrame()
        cols = ['PR Distances', 'PR Slope', 'PR Amplitude', 
                'PQ Distances', 'PQ Slope', 'PQ Amplitude',
                
                'QS Distances', 'QS Slope', 'QS Amplitude', 
                'ST Distances', 'ST Slope', 'ST Amplitude',
                
                'RT Distances', 'RT Slope', 'RT Amplitude', 
                'PS Amplitude', 'PT Amplitude', 'TQ Amplitude',
                'QR Amplitude', 'RS Amplitude',
                
                'QR Interval', 'RS Interval', 'PQ Interval', 
                'QS Interval', 'PS Interval', 'PR Interval',
                'ST Interval', 'QT Interval', 'RT Interval',
                'PT Interval']
        
        
        if table == 'Person':
            df = pd.DataFrame(columns=['ID', 'Name', 'Email', 'Phone number', 'Model'], data=fetched_data)
            
        elif table == 'Identification_ECG_Features':
            df = pd.DataFrame(columns= cols + ['Person'], data=fetched_data)
        
        elif table == 'Authentication_ECG_Features' or 'Fake_Person':
            df = pd.DataFrame(columns= cols + ['label'], data=fetched_data)
        
        return df


    '''
        this function takes the table that we want to insert data in it, and the data in a form of dataframe,
        and it convert the data to list of lists/tuples (if we want to insert many rows), 
        otherwise it takes the dataframe as it to insert it the database.
    '''
    def insert(self, table, data=[]):
        connection = sqlite3.connect('Heartizm.db')
        cursor = connection.cursor()
        
        if table == 'Person':
            command = self.insert_person_command()
            cursor.execute(command)
        
        elif table == 'Identification_ECG_Features' or table == 'Authentication_ECG_Features':
            command = self.insert_features_command(table)
            # print(data.to_numpy())
            
            cursor.executemany(command, data.to_numpy())
        
        
        connection.commit()
        connection.close()


    '''
        this function creates the necessary tables in the database such as:
        1- Person table:                         every new person sign up in the app will be stored in this table.
        2- Identification_ECG_Features table:    features that will be used on the identification will be stored in this table.
        3- Authentication_ECG_Features table:    features that will be used on the authentication will be stored in this table.
        4- Fake_Person table:                    this table will have an initial values for the main ECG features with label equals 0.
    '''
    def create(self):
        connection = sqlite3.connect('Heartizm.db')
        cursor = connection.cursor()

        cursor.execute('''
                    CREATE TABLE IF NOT EXISTS Person(
                        ID TEXT UNIQUE,
                        Name TEXT,
                        email TEXT,
                        phone_number TEXT,
                        ML_model TEXT
                    )
                    ''')

        connection.commit()

        cursor.execute('''
                    CREATE TABLE IF NOT EXISTS Identification_ECG_Features(
                        PR_Distances NUMERIC,
                        PR_Slope NUMERIC,
                        PR_Amplitude NUMERIC,
                        
                        PQ_Distances NUMERIC,
                        PQ_Slope NUMERIC,
                        PQ_Amplitude NUMERIC,
                        
                        QS_Distances NUMERIC,
                        QS_Slope NUMERIC,
                        QS_Amplitude NUMERIC,
                        
                        ST_Distances NUMERIC,
                        ST_Slope NUMERIC,
                        ST_Amplitude NUMERIC,
                        
                        RT_Distances NUMERIC,
                        RT_Slope NUMERIC,
                        RT_Amplitude NUMERIC,
                        
                        PS_Amplitude NUMERIC,
                        PT_Amplitude NUMERIC,
                        TQ_Amplitude NUMERIC,
                        QR_Amplitude NUMERIC,
                        RS_Amplitude NUMERIC,
                        
                        QR_Interval NUMERIC,
                        RS_Interval NUMERIC,
                        PQ_Interval NUMERIC,
                        QS_Interval NUMERIC,
                        PS_Interval NUMERIC,
                        PR_Interval NUMERIC,
                        ST_Interval NUMERIC,
                        QT_Interval NUMERIC,
                        RT_Interval NUMERIC,
                        PT_Interval NUMERIC,
                        Person TEXT
                    )
                    ''')
        
        connection.commit()
        
        cursor.execute('''
                    CREATE TABLE IF NOT EXISTS Authentication_ECG_Features(
                        PR_Distances NUMERIC,
                        PR_Slope NUMERIC,
                        PR_Amplitude NUMERIC,
                        
                        PQ_Distances NUMERIC,
                        PQ_Slope NUMERIC,
                        PQ_Amplitude NUMERIC,
                        
                        QS_Distances NUMERIC,
                        QS_Slope NUMERIC,
                        QS_Amplitude NUMERIC,
                        
                        ST_Distances NUMERIC,
                        ST_Slope NUMERIC,
                        ST_Amplitude NUMERIC,
                        
                        RT_Distances NUMERIC,
                        RT_Slope NUMERIC,
                        RT_Amplitude NUMERIC,
                        
                        PS_Amplitude NUMERIC,
                        PT_Amplitude NUMERIC,
                        TQ_Amplitude NUMERIC,
                        QR_Amplitude NUMERIC,
                        RS_Amplitude NUMERIC,
                        
                        QR_Interval NUMERIC,
                        RS_Interval NUMERIC,
                        PQ_Interval NUMERIC,
                        QS_Interval NUMERIC,
                        PS_Interval NUMERIC,
                        PR_Interval NUMERIC,
                        ST_Interval NUMERIC,
                        QT_Interval NUMERIC,
                        RT_Interval NUMERIC,
                        PT_Interval NUMERIC,
                        Person_ID TEXT
                    )
                    ''')
        
        connection.commit()
        
        cursor.execute('''
                    CREATE TABLE IF NOT EXISTS Fake_Person(
                        PR_Distances NUMERIC,
                        PR_Slope NUMERIC,
                        PR_Amplitude NUMERIC,
                        
                        PQ_Distances NUMERIC,
                        PQ_Slope NUMERIC,
                        PQ_Amplitude NUMERIC,
                        
                        QS_Distances NUMERIC,
                        QS_Slope NUMERIC,
                        QS_Amplitude NUMERIC,
                        
                        ST_Distances NUMERIC,
                        ST_Slope NUMERIC,
                        ST_Amplitude NUMERIC,
                        
                        RT_Distances NUMERIC,
                        RT_Slope NUMERIC,
                        RT_Amplitude NUMERIC,
                        
                        PS_Amplitude NUMERIC,
                        PT_Amplitude NUMERIC,
                        TQ_Amplitude NUMERIC,
                        QR_Amplitude NUMERIC,
                        RS_Amplitude NUMERIC,
                        
                        QR_Interval NUMERIC,
                        RS_Interval NUMERIC,
                        PQ_Interval NUMERIC,
                        QS_Interval NUMERIC,
                        PS_Interval NUMERIC,
                        PR_Interval NUMERIC,
                        ST_Interval NUMERIC,
                        QT_Interval NUMERIC,
                        RT_Interval NUMERIC,
                        PT_Interval NUMERIC,
                        label INT
                    )
                    ''')
        
        connection.commit()

        connection.close()


# model_path = 'C:\\Users\\Steven20367691\\Desktop\\new prototype 1\\'
# other_users_features = pd.read_csv('C:\\Users\\Steven20367691\\Desktop\\ecg.csv')
# other_users_features.drop('Unnamed: 0', axis=1, inplace=True)

other_users_features = pd.DataFrame()

# all_features = pd.DataFrame()
extracted_features = pd.DataFrame()
predictions = pd.DataFrame(columns=['Results'])

# create a database if it's not exists.
creation = sql_ecg()
creation.create()
other_users_features = creation.fetch('*', 'Fake_Person')

person = sql_ecg()
login_data = []

model = ExtraTreesClassifier(n_estimators=200, criterion='entropy', verbose=2)

app = Flask(__name__)
api = Api(app)




'''
# class ECG_API(Resource):
#     def post(self, file_name):
#         global extracted_features
#         original_ecg_signal = pd.read_csv(file_path+file_name)
        
#         ecg_heart = ECG(original_ecg_signal)
        
#         extracted_features = ecg_heart.feature_exctraction()
#         return ''

#     def get(self, file_name):
        
#         ExtraTree_model = joblib.load(model_path + 'Extra tree banha 2.h5')
#         extracted_features.dropna(inplace=True)
        
#         preds = ExtraTree_model.predict(extracted_features)
#         print(preds)
#         predictions = pd.DataFrame({'Results': preds})
#         return {'Results': predictions.value_counts().index[0][0]}


# api.add_resource(ECG_API, '/<string:file_name>')

'''


'''
    this function takes a json data for ECG data and convert it to dict datatype. 
    return a {dataframe of the ECG data}.
'''
def convert_json_dict(ecg_json):
    ecg_dict = {}
    for column in ecg_json.keys():
        ecg_dict[column] = list(ecg_json[column].values())
        
    ecg_df = pd.DataFrame(ecg_dict)
    return ecg_df


@app.errorhandler(400)
def handle_400_error(_error):
    return make_response(jsonify({'error': 'Bad Request'}), 400)

@app.errorhandler(401)
def handle_401_error(_error):
    return make_response(jsonify({'error': 'Unauthorized'}), 401)

@app.errorhandler(403)
def handle_403_error(_error):
    return make_response(jsonify({'error': 'Forbidden'}), 403)

@app.errorhandler(404)
def handle_404_error(_error):
    return make_response(jsonify({'error': 'Not Found'}), 404)

@app.errorhandler(408)
def handle_408_error(_error):
    return make_response(jsonify({'error': 'Request Timeout'}), 408)

@app.errorhandler(500)
def handle_500_error(_error):
    return make_response(jsonify({'error': 'Internal Server Error'}), 500)

@app.errorhandler(504)
def handle_504_error(_error):
    return make_response(jsonify({'error': 'Gateway Timeout'}), 504)



'''
########################################################################################################################
########################################################################################################################
##############################################                           ###############################################
##############################################            API            ###############################################
##############################################                           ###############################################
########################################################################################################################
########################################################################################################################
'''

########################################################################################################################
#########################################            Identification            #########################################
########################################################################################################################

'''
    this API function task is to save the ML model.
    it takes the model's name and save it with .h5 extention in the same path of the project.
'''
@app.route('/identification/save_model', methods=['POST'])
def save_model():
    global model
    
    json_data = json.loads(request.data)
    model_name = json_data['Model Name']
    
    joblib.dump(model, model_name+'.h5')
    
    return ' '


'''
    this API function task is to load the ML model.
    it takes the model's name that we want to load to use it in identification.
'''
@app.route('/identification/load_model', methods=['POST'])
def load_model():
    global model
    
    json_data = json.loads(request.data)
    model_name = json_data['Model Name']
    print(model_name)
    model = joblib.load(model_name+'.h5')
    
    return ' '


'''
    this API function task is to store the main 30 features in a database in Identification_ECG_Features table.
'''
@app.route('/identification/store', methods=['POST'])
def identification_store():
    global extracted_features
    # global all_features
    
    json_data = json.loads(request.data)
    ecg_df = convert_json_dict(json_data)
    ecg_heart = ECG(ecg_df)
    extracted_features = ecg_heart.identification_labled_feature_exctraction()
    
    creation.insert('Identification_ECG_Features', extracted_features)
    
    # if len(all_features) == 0:
    #     all_features = extracted_features
    # else:
    #     all_features = pd.concat([all_features, extracted_features])
        
    # print(all_features)
    
    return ' '


'''
    this API function task is to train a new model with the main 30 features from the Identification_ECG_Features table.
    and return the {ML model performance}.
'''
@app.route('/identification/train', methods=['GET'])
def identification_train():
    # global all_features
    global model
    
    all_features = creation.fetch('*', 'Identification_ECG_Features')
    
    df = all_features.dropna()
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = ExtraTreesClassifier(n_estimators=200, criterion='entropy', verbose=2)
    model.fit(X_train, y_train)
    # preds = ExtraTree.predict(X_test)

    # ExtraTree model
    model_preds = model.predict(X_test)
    print('accuracy_score:', accuracy_score(model_preds, y_test.values))
    print('f1_score:', f1_score(y_test.values, model_preds, average='weighted'))
    print('recall_score:', recall_score(model_preds, y_test.values, average='weighted'))
    print('precision_score:', precision_score(model_preds, y_test.values, average='weighted'))

    return jsonify({'Performance': f'- Accuracy Score: {accuracy_score(model_preds, y_test.values)}'+
                    f'\n- F1 Score: {f1_score(model_preds, y_test.values, average="weighted")}'
                    +f'\n- Recall Score: {recall_score(model_preds, y_test.values, average="weighted")}'
                    +f'\n- Precision Score: {precision_score(model_preds, y_test.values, average="weighted")}'})


'''
    this API function task is to take an ECG record from the user and exctract the main 30 features from it.
    and store them in a global variable to pass it to the ML model for prediction.
'''
@app.route('/identification', methods=['POST'])
def post():
    global extracted_features
    
    json_data = json.loads(request.data)
    ecg_df = convert_json_dict(json_data)
    ecg_heart = ECG(ecg_df)
    extracted_features = ecg_heart.feature_exctraction()
    print(extracted_features)
    
    return ' '


'''
    this API function task is to pass the main 30 features to the ML model and return the {predictions}.
'''
@app.route('/identification', methods=['GET'])
def get():
    global model
    # ExtraTree_model = joblib.load('Extra tree test 11 (97).h5')
    extracted_features.dropna(inplace=True)
    print(extracted_features)
    
    preds = model.predict(extracted_features)
    predictions = pd.DataFrame({'Results': preds}) 
    print(predictions)
    return jsonify({'Results': predictions.value_counts().index[0][0]})







########################################################################################################################
#########################################            Authentication            #########################################
########################################################################################################################

'''
    this API function task is to take a person's Name and Phone Number, 
    then get all data about that person from the database.
'''
@app.route('/authentication/login', methods=['POST', 'GET'])
def authentication_login():
    global login_data
    
    json_data = json.loads(request.data)
    
    connection = sqlite3.connect('Heartizm.db')
    cursor = connection.cursor()
         
    cursor.execute('SELECT * FROM Person WHERE Name="'+ json_data['User Name'] +'" AND phone_number="'+ json_data['Phone Number'] +'"')
    login_data = cursor.fetchall()

    connection.commit()
    connection.close()

    return jsonify({'Result':'Done'})


'''
    this API function task is to take a person's Name, Email and Phone Number,
    and insert his/her data in Person table in the database.
'''
@app.route('/authentication/new_user', methods=['POST'])
def authentication_new_user():
    global other_users_features
    global person
    
    ID = secrets.token_urlsafe(32)
    json_data = json.loads(request.data)
    person = sql_ecg(ID, json_data['User Name'], json_data['Email'], json_data['Phone Number'])
    
    connection = sqlite3.Connection('Heartizm.db')
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM Person')
    records = cursor.fetchall()
    
    records_list = pd.DataFrame(records).values
    if person.person_name in records_list:
        return jsonify({'Result':'This username already exists...'})
    else:
        other_users_features = person.fetch('*', 'Fake_Person')
        person.insert('Person')
#         print(person.person_ID, '    ', person.person_name)
    
        return jsonify({'Person ID':person.person_ID, 'Person Name': person.person_name})


'''
    this API function task is to take the ECG data file and extract the main 30 features, 
    then store them with label 1 for training in authentication task.  
'''
@app.route('/authentication/store', methods=['POST', 'GET'])
def authentication_store():
    global extracted_features
    global other_users_features
    global person
    
    print(person.person_ID, '    ', person.person_name)
    json_data = json.loads(request.data)
    ecg_df = convert_json_dict(json_data)
    ecg_heart = ECG(ecg_df)
    extracted_features = ecg_heart.authentication_labled_feature_exctraction()
    
    extracted_features['label'] = extracted_features.iloc[:, -1] 
    extracted_features.drop(0, axis=1, inplace=True)
    
    other_users_features = pd.concat([other_users_features, extracted_features])
    print(other_users_features)
    
    return jsonify({'Result':'Done'})


'''
    this API function task is to train a new model with the main 30 features.
    save the model in the database in the same row of the authenticated person.
    and return the {ML model performance}.
'''
@app.route('/authentication/train', methods=['GET'])
def authentication_train():
    global other_users_features
    global person
    
    print(person.person_ID, '    ', person.person_name)
    df = other_users_features.dropna()
    print(other_users_features)
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    ExtraTree = ExtraTreesClassifier(n_estimators=200, criterion='entropy', verbose=2)
    ExtraTree.fit(X_train, y_train)

    ExtraTree_preds = ExtraTree.predict(X_test)

    person.insert_model(ExtraTree)
    print('Saved ...')
    
    return jsonify({'Performance': f'- Accuracy Score: {accuracy_score(ExtraTree_preds, y_test.values)}'
                    +f'\n- F1 Score: {f1_score(ExtraTree_preds, y_test.values, average="weighted")}'
                    +f'\n- Recall Score: {recall_score(ExtraTree_preds, y_test.values, average="weighted")}'
                    +f'\n- Precision Score: {precision_score(ExtraTree_preds, y_test.values, average="weighted")}'})


'''
    this API function task is to take the ECG record from the user, extract the main 30 features,
    then send them to the ML model for prediction.
    
    return the {predictions}.
'''
@app.route('/authentication/authenticate_perdict', methods=['POST', 'GET'])
def predict_authenticate():
    global predictions
    # global person
    global login_data
    
    # fetched_data = person.fetch('*', 'Person')
    print(login_data[0][-1])
    
    ExtraTree_model = joblib.load(login_data[0][-1])
    
    json_data = json.loads(request.data)
    ecg_df = convert_json_dict(json_data)
    ecg_heart = ECG(ecg_df)
    extracted_features = ecg_heart.feature_exctraction()
    extracted_features.dropna(inplace=True)
    
    preds = ExtraTree_model.predict(extracted_features)
    print(preds)
    
    predictions = pd.DataFrame(columns=['Results'])
    predictions['Results'] = preds
    
    return jsonify({'Result':'Done'})


'''
    this API function task is to take the predictions and determine if the user is authenticated or not.
'''
@app.route('/authentication/authenticate_results', methods=['GET'])
def authenticate_result():
    global predictions
    
    if predictions.value_counts().index[0][0] == 0:
        return jsonify({'Result':'Not Authenticated'})
    else:
        return jsonify({'Result':'Authenticated'})


app.run(host='0.0.0.0')
