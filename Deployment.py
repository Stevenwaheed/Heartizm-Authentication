import streamlit as st
import requests
import pandas as pd
import json
import os
import webbrowser

authentication_URL = 'http://127.0.0.1:5000/authentication'
identification_URL = 'http://127.0.0.1:5000/identification'


def df_to_dict(features):
    ECG_dict = {}
    for col in features.columns:
        ECG_dict[col] = list(features[col].values)
        
    return ECG_dict


def store(your_choice):
    
    file_data = pd.read_csv(ecg_file)
    file_data.to_json('train_json_data.json')
    
    with open('train_json_data.json') as data_file:
        ecg_json = json.load(data_file)
    
    if your_choice == 'authentication':
        try:
            response = requests.post(authentication_URL+'/store', json=ecg_json, timeout=120)
            st.text("Storing Done ...")
            return response
        except requests.exceptions.ConnectionError:
            st.write('Connection Error...')
        except requests.exceptions.HTTPError:
            st.write('HTTP Error...')
        except requests.exceptions.ReadTimeout:
            st.write('Read Timeout Error...')
        except requests.exceptions.RequestException:
            st.write('Request Exception Error...')
                        
    else:
        try:
            response = requests.post(identification_URL+'/store', json=ecg_json, timeout=120)
            st.text("Storing Done ...")
            return response
        except requests.exceptions.ConnectionError:
            st.write('Connection Error...')
        except requests.exceptions.HTTPError:
            st.write('HTTP Error...')
        except requests.exceptions.ReadTimeout:
            st.write('Read Timeout Error...')
        except requests.exceptions.RequestException:
            st.write('Request Exception Error...')

    
        


menu = ["Train a model", "Identification", "Authentication (Sign Up)", "Authentication (LogIn)"]
choice = st.sidebar.selectbox("Select Option", menu)




if choice == 'Train a model':
    st.header("Train a model")
    ecg_file = st.file_uploader('Upload file', type=['CSV'])
    model_name = st.text_input('Model Name')
    
    if st.button('Store'):
        try:
            response = store('identification')
        except requests.exceptions.ConnectionError:
            st.write('Connection Error...')
        except requests.exceptions.HTTPError:
            st.write('HTTP Error...')
        except requests.exceptions.ReadTimeout:
            st.write('Read Timeout Error...')
        except requests.exceptions.RequestException:
            st.write('Request Exception Error...')
    
    if st.button('Train'):
        try:
            response = requests.get(identification_URL+'/train', timeout=120)
            st.text('Training Done ...')
            st.text(response.json()['Performance'])
        except requests.exceptions.ConnectionError:
            st.write('Connection Error...')
        except requests.exceptions.HTTPError:
            st.write('HTTP Error...')
        except requests.exceptions.ReadTimeout:
            st.write('Read Timeout Error...')
        except requests.exceptions.RequestException:
            st.write('Request Exception Error...')
        
        
    if st.button('Save model'):
        model_name_input = {'Model Name':model_name}
        try:
            response = requests.post(identification_URL+'/save_model', json=model_name_input, timeout=120)
            st.text('Saved Successfully ...')
        except requests.exceptions.ConnectionError:
            st.write('Connection Error...')
        except requests.exceptions.HTTPError:
            st.write('HTTP Error...')
        except requests.exceptions.ReadTimeout:
            st.write('Read Timeout Error...')
        except requests.exceptions.RequestException:
            st.write('Request Exception Error...')
    
        
        

    
elif choice == 'Identification':
    st.header("Identification")
    ecg_file = st.file_uploader('Upload file', type=['CSV'])
    model_name = st.text_input('Model Name')
    
    if st.button('Post'):
        file_data = pd.read_csv(ecg_file)
        file_data.to_json('json_data.json')
        
        with open('json_data.json') as data_file:
            ecg_json = json.load(data_file)

        try:
            response = requests.post(identification_URL, json=ecg_json, timeout=120)
            st.text("Done ...")
        except requests.exceptions.ConnectionError:
            st.write('Connection Error...')
        except requests.exceptions.HTTPError:
            st.write('HTTP Error...')
        except requests.exceptions.ReadTimeout:
            st.write('Read Timeout Error...')
        except requests.exceptions.RequestException:
            st.write('Request Exception Error...')

    if st.button('Load model'):
        model_name_input = {'Model Name':model_name}
        
        try:
            response = requests.post(identification_URL+'/load_model', json=model_name_input, timeout=120)
            st.text("Loaded Successfully ...")
        except requests.exceptions.ConnectionError:
            st.write('Connection Error...')
        except requests.exceptions.HTTPError:
            st.write('HTTP Error...')
        except requests.exceptions.ReadTimeout:
            st.write('Read Timeout Error...')
        except requests.exceptions.RequestException:
            st.write('Request Exception Error...')

    if st.button('Get'):
        try:
            response = requests.get(identification_URL, timeout=120)
            st.text(response.json())
        except requests.exceptions.ConnectionError:
            st.write('Connection Error...')
        except requests.exceptions.HTTPError:
            st.write('HTTP Error...')
        except requests.exceptions.ReadTimeout:
            st.write('Read Timeout Error...')
        except requests.exceptions.RequestException:
            st.write('Request Exception Error...')




elif choice == 'Authentication (Sign Up)':
    st.header("Authentication")
    
    st.subheader('User Information (Sign up)')
    
    person_name = st.text_input('Enter your name')
    person_phone_number = st.text_input('Enter your phone number')
    person_email = st.text_input('Enter your email')
    
    if st.button('Add User'):
        st.success('Added Successfully')
        info = {'User Name':person_name, 'Email': person_email, 'Phone Number':person_phone_number}
        
        try:
            response = requests.post(authentication_URL+'/new_user', json=info, timeout=120)
        except requests.exceptions.ConnectionError:
            st.write('Connection Error...')
        except requests.exceptions.HTTPError:
            st.write('HTTP Error...')
        except requests.exceptions.ReadTimeout:
            st.write('Read Timeout Error...')
        except requests.exceptions.RequestException:
            st.write('Request Exception Error...')
    
    
    st.write('---------------------------------------------------------------------')
    st.subheader('Enter your heart signals')
    ecg_file = st.file_uploader('Upload file', type=['CSV'])
    
    if st.button('Store'):
        try:
            response = store('authentication')
        except requests.exceptions.ConnectionError:
            st.write('Connection Error...')
        except requests.exceptions.HTTPError:
            st.write('HTTP Error...')
        except requests.exceptions.ReadTimeout:
            st.write('Read Timeout Error...')
        except requests.exceptions.RequestException:
            st.write('Request Exception Error...')

    if st.button('Train'):
        try:
            response = requests.get(authentication_URL+'/train', timeout=120)
            st.text('Training Done ...')
            st.text(response.json()['Performance'])
        except requests.exceptions.ConnectionError:
            st.write('Connection Error...')
        except requests.exceptions.HTTPError:
            st.write('HTTP Error...')
        except requests.exceptions.ReadTimeout:
            st.write('Read Timeout Error...')
        except requests.exceptions.RequestException:
            st.write('Request Exception Error...')


elif choice == 'Authentication (LogIn)':
    st.subheader('User Information (Login)')
    
    person_name = st.text_input('Login with your name')
    person_phone_number = st.text_input('Login with your phone number')
    
    if st.button('Login'):
        info = {'User Name':person_name, 'Phone Number':person_phone_number}
        
        try:
            response = requests.post(authentication_URL+'/login', json=info, timeout=120)
            st.text('Done ...')
        except requests.exceptions.ConnectionError:
            st.write('Connection Error...')
        except requests.exceptions.HTTPError:
            st.write('HTTP Error...')
        except requests.exceptions.ReadTimeout:
            st.write('Read Timeout Error...')
        except requests.exceptions.RequestException:
            st.write('Request Exception Error...')
    
    
    st.write('---------------------------------------------------------------------')
    ecg_auth_file = st.file_uploader('Test Upload File', type=['CSV'])
    if st.button('Authenticate (POST)'):
        
        file_data = pd.read_csv(ecg_auth_file)
        file_data.to_json('authenticate_json_data.json')
        
        with open('authenticate_json_data.json') as data_file:
            ecg_json = json.load(data_file)
        
        try:
            response = requests.post(authentication_URL+'/authenticate', json=ecg_json, timeout=120)
            st.text('Done ...')
        except requests.exceptions.ConnectionError:
            st.write('Connection Error...')
        except requests.exceptions.HTTPError:
            st.write('HTTP Error...')
        except requests.exceptions.ReadTimeout:
            st.write('Read Timeout Error...')
        except requests.exceptions.RequestException:
            st.write('Request Exception Error...')
        
    if st.button('Authenticate (GET)'):
        try:
            response = requests.get(authentication_URL+'/authenticate', timeout=120)
            if response.json()['Result'] == 'Authenticated':
                st.success('Welcome you successfully can access this account.')
                # webbrowser.open_new_tab('https://www.facebook.com/?_rdc=2&_rdr')
                st.markdown("[![Foo](https://upload.wikimedia.org/wikipedia/commons/f/fb/Facebook_icon_2013.svg)](https://www.facebook.com/?_rdc=2&_rdr)")
                # sleep for two sec
                
            elif response.json()['Result'] == 'Not Authenticated':
                st.error('You\'re not authenticated.')
                
            # st.text(response.json()['Result'])
            
        except requests.exceptions.ConnectionError:
            st.write('Connection Error...')
        except requests.exceptions.HTTPError:
            st.write('HTTP Error...')
        except requests.exceptions.ReadTimeout:
            st.write('Read Timeout Error...')
        except requests.exceptions.RequestException:
            st.write('Request Exception Error...')