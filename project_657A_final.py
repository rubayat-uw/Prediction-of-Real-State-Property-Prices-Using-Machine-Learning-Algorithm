#import some necessary librairies

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
import scipy.stats as stats
from scipy.stats import norm, skew #for some statistics

#Now let's import and put the train and test datasets in  pandas dataframe
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_raw=train
test_raw=test

ytemp=train["SalePrice"]
ytemp=pd.DataFrame(ytemp)
#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

#Outlier
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

#Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

sns.distplot(train['SalePrice'] , fit=norm);
# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train["SalePrice"] = np.log1p(train["SalePrice"])

#Check the new distribution 
sns.distplot(train['SalePrice'] , fit=norm);
#Check the new distribution 
#sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)

#Missing Data
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)

f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)

#Correlation map to see how features are correlated with SalePrice
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)

#Imputing missing values
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
    
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
    
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
    
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

all_data = all_data.drop(['Utilities'], axis=1)

all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])


all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

#Check remaining missing values if any 
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()


#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)


from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data: {}'.format(all_data.shape))

# Adding total sqfootage feature 
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data_raw=all_data
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)

skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)
    
#all_data[skewed_features] = np.log1p(all_data[skewed_features])
    
all_data = pd.get_dummies(all_data)
print(all_data.shape)

train = all_data[:ntrain]
test = all_data[ntrain:]

#Modelling
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dropout


####Normalize using min max scalar
scaler = MinMaxScaler()
train1 = scaler.fit_transform(train)
y_train1=pd.DataFrame(y_train)
test1 = scaler.fit_transform(y_train1)

###NN
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
X_train, X_test, y_train, y_test = train_test_split(train1, test1, test_size=0.25, random_state=seed)
from keras.models import Sequential

model = Sequential()
model.add(Dense(100, input_dim=220, activation='linear'))
#model.add(Dropout(0.25))
#model.add(Dense(3, activation='linear'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1,  activation='linear'))
# Compile model
model.compile(loss='mse' , optimizer='rmsprop')
# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_test,y_test),epochs=1)

##predict
y_pred_nn = model.predict(X_test)

# summarize history for loss
fig, ax = plt.subplots()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
#fig, ax = plt.subplots()


##Re scale the pred and test
y_test_de_trans=scaler.inverse_transform(y_test)
y_pred_nn_de_trans=scaler.inverse_transform(y_pred_nn)
###reverse the log value
y_test_exp=np.exp(y_test_de_trans)-1
y_pred_nn_exp=np.exp(y_pred_nn_de_trans)-1

##convert into list for slope and intercept

x_slope = [i[0] for i in y_test_exp]
x_intercept = [i[0] for i in y_pred_nn_exp]
slope, intercept = np.polyfit(x_slope, x_intercept, 1)
abline_values_nn = [slope * i + intercept for i in x_slope]

######RF
from sklearn.ensemble import RandomForestRegressor
regressor_forest = RandomForestRegressor(n_estimators = 150, random_state = 0, min_samples_leaf = 5,
                                    max_features = 'auto')
regressor_forest.fit(X_train, y_train)
y_pred_rf = regressor_forest.predict(X_test)

##Re scale the pred and test
#y_testit=scaler.inverse_transform(y_test)
##reshape to 2d 
y_pred_rf=y_pred_rf.reshape(-1,1)
y_pred_rf_de_trans=scaler.inverse_transform(y_pred_rf)
####reverse the log value
#
y_pred_rf_exp=np.exp(y_pred_rf_de_trans)-1
#
###convert into list for slope and intercept
#
#x1 = [i[0] for i in y_test_exp]
x_intercept = [i[0] for i in y_pred_rf_exp]
slope, intercept = np.polyfit(x_slope, x_intercept, 1)
abline_values_rf = [slope * i + intercept for i in x_slope]

##SVR

from sklearn.svm import SVR
svr_rbf = SVR(C=1.0, gamma=0.0001,
    kernel='rbf')
#svr_rbf.fit(X_train, y_train)
y_pred_svr =svr_rbf.fit(X_train, y_train).predict(X_test)

#Re scale the pred and test
#y_testit=scaler.inverse_transform(y_test)
#reshape to 2d 
y_pred_svr=y_pred_svr.reshape(-1,1)
y_pred_svr_de_trans=scaler.inverse_transform(y_pred_svr)
####reverse the log value
#
y_pred_svr_exp=np.exp(y_pred_svr_de_trans)-1
#

###convert into list for slope and intercept
#
#x1 = [i[0] for i in y_test_exp]
x_intercept = [i[0] for i in y_pred_svr_exp]
slope, intercept = np.polyfit(x_slope, x_intercept, 1)
abline_values_svr= [slope * i + intercept for i in x_slope]
######XGBRegressor
import xgboost as xgb
regr_xgb = xgb.XGBRegressor(
                 colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.01,
                 max_depth=4,
                 min_child_weight=1.5,
                 n_estimators=30000,                                                                  
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.2,
                 seed=42,
                 silent=1)

best_alpha = 0.00098
regr_xgb.fit(X_train, y_train)
##predict
y_pred_xgb= regr_xgb.predict(X_test)
#reshape to 2d 
y_pred_xgb=y_pred_xgb.reshape(-1,1)
#Re scale the pred and test
y_pred_xgb_de_trans=scaler.inverse_transform(y_pred_xgb)
####reverse the log value
#
y_pred_xgb_exp=np.exp(y_pred_xgb_de_trans)-1


###convert into list for slope and intercept
#
#x1 = [i[0] for i in y_test_exp]
x_intercept = [i[0] for i in y_pred_xgb_exp]
slope, intercept = np.polyfit(x_slope, x_intercept, 1)
abline_values_xgb= [slope * i + intercept for i in x_slope]

fig=plt.figure(figsize=(7,7))
fig.subplots_adjust(hspace=.4)
fig.subplots_adjust(wspace=.4)
#fig.suptitle("Train vs Predict", fontsize=16)

##plot NN
ax= fig.add_subplot(2,2,1)
plt.plot(y_test_exp,y_pred_nn_exp,'o')
plt.plot(y_test_exp, abline_values_nn, '-')
plt.xlabel('Predictated Sales Price', fontsize=13)
plt.ylabel('Test Sale Price', fontsize=13)
plt.title('Neural Network Regressor')


##plot RF
ax= fig.add_subplot(2,2,2)
plt.plot(y_test_exp,y_pred_rf_exp,'o')
plt.plot(y_test_exp, abline_values_rf, '-')
plt.xlabel('Predictated Sales Price', fontsize=13)
plt.ylabel('Test Sale Price', fontsize=13)
plt.title('Random Forest Regressor')


##plot SVR
ax= fig.add_subplot(2,2,3)
plt.plot(y_test_exp,y_pred_svr_exp,'o')
plt.plot(y_test_exp, abline_values_svr, '-')
plt.xlabel('Predictated Sales Price', fontsize=13)
plt.ylabel('Test Sale Price', fontsize=13)
plt.title('Support Vector Regression')

##plot XGB
ax= fig.add_subplot(2,2,4)
plt.plot(y_test_exp,y_pred_xgb_exp,'o')
plt.plot(y_test_exp, abline_values_xgb, '-')
plt.xlabel('Predictated Sales Price', fontsize=13)
plt.ylabel('Test Sale Price', fontsize=13)
plt.title('XGBOOST')
plt.show()

####predict vs test plot

###test vs predict plot
fig=plt.figure(figsize=(10,9))
fig.subplots_adjust(hspace=.7)
#fig.subplots_adjust(wspace=.5)
fig.suptitle("Test vs Predict", fontsize=16)

ax= fig.add_subplot(4,1,1)

plt.plot(y_test_exp,'red',label='Test')
plt.plot(y_pred_svr_exp,label='SVR Predict')
plt.xlabel('Samples')
plt.ylabel('Sales Price')
plt.legend()

ax= fig.add_subplot(4,1,2)
###test vs predict plot
plt.plot(y_test_exp,'red',label='Test')
plt.plot(y_pred_rf_exp,label='RF Predict')
plt.xlabel('Samples')
plt.ylabel('Sales Price')
plt.legend()

###test vs predict plot
ax= fig.add_subplot(4,1,3)
plt.plot(y_test_exp,'red',label='Test')
plt.plot(y_pred_nn_exp,label='NN Predict')
plt.xlabel('Samples')
plt.ylabel('Sales Price')
plt.legend()

ax= fig.add_subplot(4,1,4)
plt.plot(y_test_exp,'red',label='Test')
plt.plot(y_pred_xgb_exp,label='XGBOOST Predict')
plt.xlabel('Samples')
plt.ylabel('Sales Price')
plt.legend()
plt.show()


def rmse(y_true, y_pred):
    rmse_val=np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse_val

rmse_nn=rmse(y_test,y_pred_nn)
rmse_rf=rmse(y_test,y_pred_rf)
rmse_svr=rmse(y_test,y_pred_svr)
rmse_xgb=rmse(y_test,y_pred_xgb)

#Model comparison
model_comparison = pd.DataFrame({
        'Model'          : ['Neural Network','Random Forest','SVR','XGBOOST'],
        'RMSE' : [rmse_nn, rmse_rf, rmse_svr, rmse_xgb]
#        'Training_Score'  : [nn_score_train, rf_score_train, svr_score_train, xgb_score_train],
#        'Testing_Score'  : [nn_score_test, rf_score_test, svr_score_test, xgb_score_test]
    })
model_comparison.sort_values(by='RMSE', ascending=True)


#plt.plot(y_test_exp[1:50],'red',label='Test')
#plt.plot(y_pred_xgb_exp[1:50],label='XGBOOST Predict')
#plt.xlabel('Samples')
#plt.ylabel('Sales Price')
#plt.legend()
#plt.show()
#

#n_folds = 5

#def rmsle_cv(model):
#    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
#    rmse= np.sqrt(-cross_val_score(model, train.values, y_train1, scoring="neg_mean_squared_error", cv = kf))
#    return(rmse)
#score = rmsle_cv(svr_rbf)
#print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
