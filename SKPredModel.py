import pandas as pd
import re
#import pickle
import xgboost as xgb


class SKPredModel:
    def __init__(self, model_path: str):
        """
        Here you initialize your model
        """
        regexp_time_list = [
            r'(?P<days>.+?(?=-))-(?P<hours>.+?(?=:)):(?P<minutes>.+?(?=:)):(?P<seconds>\d+)',
            r'(?P<hours>.+?(?=:)):(?P<minutes>.+?(?=:)):(?P<seconds>\d+)'
        ]

        #self.regressor = pickle.load(open(pickle_dump_path, 'rb'))

        self.SEED = 314159265

        self.regressor = xgb.XGBRegressor(booster = 'gbtree',
                           colsample_bytree = 1.0,
                           learning_rate = 0.025,
                           gamma = 1.0,
                           max_depth = 13,
                           min_child_weight = 3,
                           n_estimators = 263,
                           n_jobs = 16,
                           objective = 'reg:squarederror',
                           random_state = self.SEED,
                           subsample = 0.7,
                           tree_method = 'exact'
                        )
        self.regressor.load_model(model_path)

        self.compiled_regexps = [re.compile(regexp) for regexp in regexp_time_list]

    def convert_time_to_seconds(self, element):
        for rcompile in self.compiled_regexps:
            rsearch = rcompile.search(element)
            if rsearch:
                try:
                    return (int(rsearch.group('days')) * 24 + int(rsearch.group('hours'))) * 3600 + int(rsearch.group('minutes')) * 60 + int(rsearch.group('seconds'))
                except:
                    return int(rsearch.group('hours')) * 3600 + int(rsearch.group('minutes')) * 60 + int(rsearch.group('seconds'))

    def prepare_df(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Here you put any feature generation algorithms that you use in your model

        :param test_df:
        :return: test_df extended with generated features
        """
        test_df.fillna(0, inplace=True)
        test_df.drop(columns=['Start', 'JobName'], inplace=True)
        test_df['Timelimit'] = test_df['Timelimit'].apply(self.convert_time_to_seconds)
        test_df['Elapsed'] = test_df['Elapsed'].apply(self.convert_time_to_seconds)
        test_df['Area'].replace({'geophys': 0,
            'radiophys': 1,
            'phys': 2,
            'bioinf': 4,
            'mach': 5,
            'biophys': 6,
            'it': 7,
            'mech': 8,
            'energ': 9,
            'astrophys': 10}, inplace=True)
        test_df['Area'].astype(int)
        test_df['Partition'].replace({'tornado': 0,
            'g2': 1,
            'cascade': 2,
            'tornado-k40': 3,
            'nv': 4}, inplace=True)
        test_df['Partition'].astype(int)
        test_df['State'].replace({
            'COMPLETED': 0,
            'FAILED': 1,
            'TIMEOUT': 2,
            'NODE_FAIL': 3,
            'OUT_OF_MEMORY': 4
        }, inplace=True)
        # test_df['State'].replace(r'(CANCELLED.+)|(CANCELLED)', 5, regex=True, inplace=True)
        # test_df['State'].astype(int)
        # test_df['ExitCode'].replace(r':','', regex=True, inplace=True)
        # test_df['ExitCode'] = test_df.ExitCode.astype(int)
        # test_df['ExitCode'].astype(int)
        test_df['Submit'] = pd.to_datetime(test_df['Submit'])
        test_df['Month'] = test_df['Submit'].dt.month
        test_df.drop(columns=['Submit'], inplace=True)

        ### Апостериорно получаемые фичи
        test_df.drop(columns=['State'], inplace=True)
        test_df.drop(columns=['ExitCode'], inplace=True)

        test_df.drop(columns=['Elapsed'], inplace=True)

        test_df.reset_index(inplace=True, drop=True)

        # mean_elapsed = pd.DataFrame(test_df[test_df['State'] == 0].groupby('UID').mean()['Elapsed'])
        # mean_elapsed.columns = ['mean_elapsed']
        # mean_elapsed.reset_index(inplace=True)
        # test_df = test_df.merge(mean_elapsed, on='UID', how='left')
        # succeded_task_count = pd.DataFrame(test_df[test_df['State'] == 0].groupby('UID').count()['State'])
        # succeded_task_count.columns = ['succeded_task_count']
        # succeded_task_count.reset_index(inplace=True)
        # failed_task_count = pd.DataFrame(test_df[test_df['State'] == 1].groupby('UID').count()['State'])
        # failed_task_count.columns = ['failed_task_count']
        # failed_task_count.reset_index(inplace=True)
        # timeout_task_count = pd.DataFrame(test_df[test_df['State'] == 2].groupby('UID').count()['State'])
        # timeout_task_count.columns = ['timeout_task_count']
        # timeout_task_count.reset_index(inplace=True)
        # tasks = succeded_task_count.merge(failed_task_count, on='UID', how='left')
        # tasks = tasks.merge(timeout_task_count, on='UID', how='left')
        # tasks.fillna({'succeded_task_count': 0, 'failed_task_count': 0, 'timeout_task_count': 0}, inplace=True)
        # tasks['succeded_task_proportion'] = tasks['succeded_task_count'] / (tasks['succeded_task_count'] + tasks['failed_task_count'] + tasks['timeout_task_count'])
        # test_df = test_df.merge(tasks, on='UID', how='left')

        return test_df

    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        """
        Here you implement inference for your model

        :param test_df: dataframe to predict
        :return: vector of estimated times in milliseconds
        """
        for key in  self.regressor.get_booster().feature_names:
            assert key in test_df.keys(), f"{key} column missed in test_df"

        predictions = self.regressor.predict(test_df)
        
        return predictions


# Usage example
'''
test_df = pd.read_csv('data/train_w_areas_st_till_june.csv',index_col=0)

model = SKPredModel('SKPred_xgboost_saved_model.json')
prepared = model.prepare_df(test_df.copy())
predictions = model.predict(prepared)
'''
