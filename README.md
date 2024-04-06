# Datamining 2024-10 Indicators of Heart Disease

Welcome to the Datamining 2024-10 project! This project aims to analyze key indicators of heart disease using data mining techniques.

## Team Members
Meet the team members working on this project:


| <img src="assets/a.jpg" alt="Jesús Antonio Zuluaga Moreno" width="200"/> | **Jesús Antonio Zuluaga Moreno** |
| --- | --- |
| *Role*: Mathematics enthusiast | *Bio*: 7th-semester mathematics student at Universidad del Norte, passionate about physics and mathematics. |

| <img src="assets/b.jpg" alt="Diego Linero Ramirez" width="200"/> | **Diego Linero Ramirez** |
| --- | --- |
| *Role*: Technology enthusiast | *Bio*: 9th-semester Systems engineering student at Universidad del Norte, passionate about innovation and technology. |

| <img src="assets/c.jpg" alt="María Valentina Páez Romero" width="200"/> | **María Valentina Páez Romero** |
| --- | --- |
| *Role*: Technology enthusiast | *Bio*: Systems engineering student at Universidad del Norte, passionate about technology and always ready to face new challenges. |

| <img src="assets/d.jpg" alt="Sebastian Barandica Oquendo" width="200"/> | **Sebastian Barandica Oquendo** |
| --- | --- |
| *Role*: Problem-solving enthusiast | *Bio*: 9th-semester systems engineering at Universidad del Norte,, lover of mathematics and problem-solving. |

| <img src="assets/e.jpg" alt="Andrea Carolina Donado Porto" width="200"/> | **Andrea Carolina Donado Porto** |
| --- | --- |
| *Role*: Versatile learner | *Bio*: Systems engineering student at Universidad del Norte, enjoys reading, thinking about everyday problems, and learning a little about everything, applying it to technology. |

## Rationale Behind Dataset Selection

**Heart disease** is a significant cause of mortality globally, and the dataset, sourced from the [**2022 annual CDC survey**](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease) of **400k+ adults**, provides invaluable insights into various factors contributing to heart disease prevalence. Our intention with utilizing this dataset is to develop a predictive model that can aid in identifying individuals at risk of heart disease. This is particularly significant in regions like Colombia, where sufficient data and datasets regarding cardiovascular diseases may be lacking. In [Colombia, cardiovascular diseases are the leading cause of mortality](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4705360/), with a significant number of these events being preventable. Given the similarity in risk factors and health patterns, using the CDC dataset on key indicators of heart disease in American adults could provide valuable **insights** to inform preventive and public health strategies in Colombia. By comparing and contrasting demographic and health data between both populations, the research and recommendations could be more effectively tailored, addressing the specific needs and characteristics of the Colombian population. Furthermore, the dataset's richness is evident in its offering of **40 variables** for observation, coupled with over **400k+ samples**.


## Visual Analysis Of Key Heart Disease Indicators: Exploring the BRFSS Dataset
In our analysis, we employed various visualization techniques with the BRFSS dataset:
### Correlation Heatmap:
We created a correlation heatmap to explore relationships between different variables, particularly those related to key indicators of heart disease. This heatmap provided a quick overview of positive or negative associations among the selected variables.
<p aling= "center">
<img src="assets/co.png" alt="" width="500"/>
</p>

### Countplot by Gender: 
We generated a countplot to visualize the distribution of heart disease cases based on gender. This type of plot allowed us to determine whether heart disease is more prevalent in males or females, offering a general perspective on the prevalence in the surveyed population.
<p aling= "center">
<img src="assets/gender.png" alt="" width="500"/>
</p>

### Age-Grouped Heart Attacks by Gender:
We grouped heart attack cases by age and gender to gain a more detailed understanding of how the incidence of heart disease varies across different demographic groups. This helped us identify potential patterns or significant differences in prevalence between males and females in different age brackets.
<p aling= "center">
<img src="assets/FandM.png" alt="" width="500"/>
</p>

### Associations Among Cardiovascular Events:Insights into Heart Attacks, Angina, and Strokes" 
Additionally, we created another visualization to demonstrate that individuals experiencing heart attacks also tend to suffer from angina and strokes. This graphic provided insights into the co-occurrence of these cardiovascular events, highlighting potential common risk factors or associations within the dataset."
<p aling= "center">
<img src="assets/comparacionAnginaStroke.png" alt="" width="1000"/>
</p>

## Our Hypothesis
Based on key indicators of heart disease, we could assert that there is a significant relationship between the prevalence of heart disease and factors such as physical health days, sex, mental health days, sleep hours, history of heart attacks, body mass index (BMI), and pre-existing conditions such as high blood pressure, high cholesterol, smoking habits, and diabetes in the adult population surveyed in the BRFSS dataset.

Another one we want to explore is:" The number of physical health days experienced in the last 30 days is associated with an increased likelihood of having had a heart attack."

# Model Selection 1: Random Forest Classifier
## Objectives for estimation
<ul>
  <li>Test the hypothesis that there is a significant relationship between the prevalence of heart disease and the selected factors in the adult population.</li>
  <li>Use random forest classifier to estimate the strength of association between these indicators and the likelihood of having heart disease.</li>
  <li>Assess the performance of the random forest classifier utilizing the BRFSS dataset to identify individuals at risk of heart disease, targeting a minimum accuracy of 0.85.</li>
</ul>

## Details about model selection
Whe choose Random for our first model selection. 
Firstly, its ability to mitigate overfitting surpasses that of individual decision trees, crucial for ensuring our model's generalization to unseen data. 
Additionally, given the imbalanced nature of our dataset, where instances of individuals with and without heart disease may vary significantly, random forest's equitable treatment of each class during training enhances its predictive accuracy. 
With over 30 input variables, random forest adeptly handles high-dimensional data without necessitating dimensionality reduction techniques.
Lastly, its robustness to outliers and missing values further reinforces its suitability for our analysis, ensuring reliable predictions even in the presence of data imperfections. 
# Hyperparameter
We choose a random state of 42  and also limited the max_depth to 3 for visualisation purpose.
# Preprocessing and visualisation
<pre>
<code>

dfHeart = pd.read_csv('heart_2022_no_nans.csv').sample(frac=0.1, random_state=42)
labels = [column for column in dfHeart.columns if dfHeart[column].dtype == 'O']

dfHeart.loc[dfHeart['HadHeartAttack'] == 'No', 'HadHeartAttack'] = 0
dfHeart.loc[dfHeart['HadHeartAttack'] == 'Yes', 'HadHeartAttack'] = 1

X = dfHeart.drop('HadHeartAttack', axis=1)
y = dfHeart['HadHeartAttack'].astype('int')

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
...
clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
</code>
</pre>
It starts with data preprocessing, including handling categorical variables through one-hot encoding and encoding target labels to numeric values. The dataset is then split into training and testing sets. Afterward, a pipeline is constructed, incorporating data preprocessing steps and the Random Forest classifier. The model is trained on the training data and evaluated on the test data, with accuracy as the metric. 
![descarga (4)](https://github.com/AnwiiD/DataMining-Heart-Failure/assets/78710847/a23bdd13-2641-4ebf-b874-04db47dbfe38)
## Validation methods and the metrics employed.
<pre>
  <code>
  from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
  </code>
</pre>
Precisión del modelo: 0.9427

Precisión: 0.5000

Recall: 0.0745

F1-score: 0.1296

In this project, we employed cross-validation, a technique used to assess the performance of a model and avoid overfitting. We also ensured proper handling of both categorical and numerical columns during preprocessing to optimize model training. 

## Conclusions
Despite achieving a model precision of 94.27%, indicating a relatively high accuracy in predicting heart attacks, further evaluation reveals areas for improvement. While precision seems good at first glance, other metrics such as recall and F1-score offer a more comprehensive picture. The recall score, which measures the ability of the model to identify all relevant instances, is notably low at 7.45%, indicating that the model misses a significant portion of actual heart attack cases. Similarly, the F1-score, which balances precision and recall, is relatively low at 12.96%. These metrics underscore the need for a more balanced model that doesn't just prioritize accuracy but also effectively identifies positive cases. Additionally, the 50% precision indicates that half of the predicted positive cases were actually negative, suggesting room for improvement in the model's ability to correctly classify instances.

## Our Colab 
In this collaborative Google Colab notebook, we are actively processing and analyzing the key indicators of heart disease using the Behavioral Risk Factor Surveillance System (BRFSS) dataset.  
#### Link: [Colab](https://colab.research.google.com/drive/1oqJ3N0Pkx_IKPrnkxjWtKydzer5bo5US?usp=sharing)
