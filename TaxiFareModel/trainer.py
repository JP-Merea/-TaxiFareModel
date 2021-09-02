# imports

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from TaxiFareModel.encoders import DistanceTransformer
from TaxiFareModel.encoders import TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import clean_data
from TaxiFareModel.data import get_data
from sklearn.model_selection import train_test_split 

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
        ('dist_trans', DistanceTransformer()),
        ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())])
        return pipe

    def run(self):
        """set and train the pipeline"""
        self.pipeline = self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)
        return rmse 


if __name__ == "__main__":
    df= get_data()
    cd = clean_data(df)
    y = cd["fare_amount"]
    X = cd.drop("fare_amount", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.15)
    trainer= Trainer(X_train,y_train)
    trainer.run()
    root_mse= trainer.evaluate(X_test,y_test)
    print(root_mse)
    
    
    #df = get_data()
    #y = df["fare_amount"]
    #X = df.drop("fare_amount", axis=1)
    #X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.15)
    #pipeline = Trainer.set_pipeline(None)
    #train = Trainer.run(df)
    #rmse = train.evaluate(pipeline, X_test, y_test)


# get data
    #df = get_data()
    #print (df)
    # clean data
    #cd = clean_data(df)
    #print (clean_data(df))
    # set X and y
    #pipe = Trainer.set_pipeline(cd)
    #print (Trainer.set_pipeline(cd))
    # hold out
    #train = pd.read_csv('../raw_data')
    # train
    #train= Trainer.run(pipe)
    #print (Trainer.run(pipe))
    # evaluate
    #print(Trainer.evaluate(train))
    # store the data in a DataFrame