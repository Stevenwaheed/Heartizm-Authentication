<h1 align="center" id="title">Heartizm-Authentication</h1>

<p id="description">Using ECG Heart Biometrics to identify people through their unique heart wave signals. Heart biometrics are preferable for continuous authentication due to their capability of providing a new biometric sample periodically. ECG signals are recorded through smart devices and sent to our cloud database applying dedicated Machine Learning algorithms and giving the user feedback on the success or failure authentication process.</p>

  
  
<h2>🧐 Features</h2>

Here're some of the project's best features:

*   Easy to Use
*   Use the best filters to remove the noise from your ECG signal
*   Use Machine Learning to predict your heart signal
*   Use your heart signal like your fingerprint to lock your system
*   Use a local database to store ecg signals for each person

<h2>🛠️ Installation Steps:</h2>

<p>1. Clone the repository</p>

```
git clone "https://github.com/Stevenwaheed/Heartizm-Authentication.git"
```

<p>2. Install npm modules</p>

```
pip install -r requirements.txt
```

<p>3. Start the server using below command</p>

```
py ECG.py
```

<p>4. Start running of streamlit website using below command</p>

```
streamlit run Deployment.py
```

  
  
<h2>💻 Built with</h2>

Technologies used in the project:

*   heartpy
*   neurokit
*   pandas
*   numpy
*   scikit-learn
*   Flask
*   Flask\_RESTful
*   streamlit
