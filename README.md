## DataSet
### Bank Marketing
https://www.kaggle.com/henriqueyamahata/bank-marketing
### World Development Indicators
https://www.kaggle.com/worldbank/world-development-indicators
<br><br>
## Classification
![classifier](https://user-images.githubusercontent.com/44195740/70368446-e5351b00-18ed-11ea-816d-df63fc26c515.png)
<br><br>
## Clustering
![cluster](https://user-images.githubusercontent.com/44195740/70368447-e6fede80-18ed-11ea-98cb-d725b8efe4f6.png)
<br><br>
## Introduction of Directories and Files
-	MachineLearningTermProject(Directory)
    -	Bank(classification)(Directory)
        -	Bank_knn.py: K-Nearest Neighbor Classification
        -	Bank_logistic.py: Logistic Regression
        -	Bank_naive: Naïve Bayesian Classification
        -	Bank_svm: Support Vector Machine Classification
        -	Bank_voting: Voting Classification (Ensemble)
        -	Bank_xgb: XGBoost Classification + XGBoost Ensemble
        -	EDA_bank: Data analysis source file
        -	EDA_correlation: To make Pearson correlation files
        -	Prep_data: To make preprocessed data
        -	Prep_scale: To make scaled data
    -	Data(Directory)
        -	Bankmarketing(Directory)
            -	Output
                -	GridSearchCV_1~2.out: Best parameter with GridSearchCV, Server
            -	*.csv
                -	‘bank-additional-full.csv’: Original dataset
                -	‘prep’: preprocessed data (delete rows which have unknown data)
                -	‘res’: Resampled data
                -	‘SCALER’: maxabs, minmax, robust, std
                -	Bank_correlation.xlsx: Pearson correlation
        -	World_development_indicators(Directory)
            -	Pivot_indicator
            -	*.csv: pivot data per keyword(topic)
            -	Prophet(Directory)
            -	*.csv: pivot data per keyword(topic)(resampled)
            -	Country.csv: Original dataset
            -	Country_Income_Level.csv
            -	CountryNotes.csv: Original dataset
            -	Footnotes.csv: Original dataset
            -	Income_level.csv: Income_level(preprocessed data)
            -	Indicators.csv: Original dataset
            -	OGHIST.xls: Income_level indicator (Additional dataset)
            -	Series.csv: Original dataset
            -	SeriesNotes.csv: Original dataset
            -	World_GNI_anually.xlsx: World GNI per year
    -	Indicators(clustering)(Directory)
        -	Ipython_files(Directory)
        -	*.jpg: Clustering Images
        -	result(Directory)
            -	*.json: Output files
            -	*.out: Output files
        -	income_level.py: To make income level each country per years
        -	ipython.html: Output files(clustering)
        -	keyword.py: To make keyword from indicators
        -	prep_prophet.py: To make preprocessed data for WiseProphet
        -	TP_2: To make cluster from data
        -	Visualization.py: To visualize clustering
