'''

Visualizamos la variable Clase
'''
survived_class = train_df[train_df['Survived']==1]['Pclass'].value_counts(sort=False)
dead_class = train_df[train_df['Survived']==0]['Pclass'].value_counts(sort=False)
class_df = pd.DataFrame([survived_class,dead_class],index=['Survived','Dead'])
class_df.plot(kind='bar', stacked=True)
plt.xlabel('Class') plt.ylabel('Numberof passengers') plt.title('Survivalrate by class') st.pyplot() 

'''
Visualizamos la variable Sexo
'''
survived_sex = train_df[train_df['Survived']==1]['Sex'].value_counts(sort=False)
dead_sex = train_df[train_df['Survived']==0]['Sex'].value_counts(sort=False)
sex_df = pd.DataFrame([survived_sex,dead_sex],index=['Survived','Dead'])
sex_df.plot(kind='bar', stacked=True) plt.xlabel('Sex') plt.ylabel('Number ofpassengers') plt.title('Survival rate by sex') st.pyplot()


'''
Visualizamos la importancia de las variables
'''
survived_embark =
train_df[train_df['Survived']==1]['Embarked'].value_counts(sort=True)
dead_embark = train_df[train_df['Survived']==0]['Embarked'].value_counts(sort=False)
st.write('Visualización del modelo y datos') st.write('Importancia de características en
el modelo') st.bar_chart(importances) 



'''
Visualizamos la variable Embarked
'''
survived_embark =
train_df[train_df['Survived']==1]['Embarked'].value_counts(sort=False)
dead_embark = train_df[train_df['Survived']==0]['Embarked'].value_counts(sort=False)
embark_df =
pd.DataFrame([survived_embark,dead_embark],index=['Survived','Dead'])
embark_df.plot(kind='bar', stacked=True) plt.xlabel('Embarked')
plt.ylabel('Number of passengers') plt.title('Survival rate by port of
embarkation') st.pyplot() 

Importamos librerías
'''
import streamlit as st
import pandas as pd
import numpy as np
imp'''
Importamos librerías
'''
import pandas import numpy from sklearn.ensemble
import RandomForestClassifier from sklearn.metrics
import accuracy_score from sklearn.model_selection
import train_test_split
st.'''
Cargamos los datos
'''
train_df = pd.read_csv('data/train.csv')
tes

Generamos Variables dependientes e independientes
'''
variables=['Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare', 'CabinBool', 'Embarked_C',
'Embarked_S', 'Embarked_Q']
X = train_df[variables]
y = train_df['Survived']

'''
Creamos el modelo
'''
X = train_df[['Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare', 'CabinBool', 'Embarked_C',
'Embarked_S', 'Embarked_Q]]
y = train_df[[Survived']]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)
# Developer

'''Hacemos predicciones'''
model.fit(X_train, y_val) y_pred =
model.predict(X_val) accuracy =
accuracy_score(y_val, y_pred)
importances =
pd.DataFrame({'feature':X_train.columns,'importance':np.round(model.feature_importa
n ces_,3)}) importances =
importances.sort_values('importance',ascending=False).set_index('feature')

'''
Visualizamos la variable Age
'''
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80] train_df['AgeGroup'] = pd.cut(train_df['Age'],
bins) survived_age =
train_df[train_df['Survived']==1]['AgeGroup'].value_counts(sort=False) dead_age =
train_df[train_df['Survived']==0]['AgeGroup'].value_counts(sort =False) age_df =
pd.DataFrame([survived_age,dead_age],index=['Survived','Dead'])
age_df.plot(kind='bar', stacked=True) plt.xlabel('Age Group') plt.ylabel('Number of
passengers')
plt.title('Distribution of passengers by age and survival') st.pyplot()

