'''
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