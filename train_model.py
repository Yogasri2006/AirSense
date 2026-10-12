import os,json,joblib,numpy as np,pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV

BASE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(BASE,'dataset','PRSA_Data_Aotizhongxin_20130301-20170228.csv'); OUT=os.path.join(BASE,'outputs'); MOD=os.path.join(BASE,'model'); os.makedirs(OUT,exist_ok=True); os.makedirs(MOD,exist_ok=True)
TARGET='PM2.5'; FEATURES=['PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM','wd','month','hour']; NUM=FEATURES[:-3]+['month','hour']; CAT=['wd']

def season(m): return 'Winter' if m in [12,1,2] else 'Spring' if m in [3,4,5] else 'Summer' if m in [6,7,8] else 'Autumn'
def prep():
    if not os.path.exists(DATA): raise FileNotFoundError(f'Put the dataset at: {DATA}')
    df=pd.read_csv(DATA); df.columns=df.columns.str.strip()
    nums=['PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM','year','month','day','hour']
    for c in nums:
        if c in df: df[c]=pd.to_numeric(df[c],errors='coerce')
    for c in ['PM2.5','PM10','SO2','NO2','CO','O3','RAIN','WSPM']:
        if c in df: df.loc[df[c]<0,c]=np.nan
    df=df.drop_duplicates(); df['datetime']=pd.to_datetime(df[['year','month','day','hour']],errors='coerce'); df=df.dropna(subset=['datetime','PM2.5']).sort_values('datetime').reset_index(drop=True)
    for c in ['PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM']:
        df[c]=df[c].interpolate(limit_direction='both').fillna(df[c].median())
    df['wd']=df['wd'].astype('string').fillna('Unknown').replace(['','nan','<NA>'],'Unknown')
    df['season']=df.month.map(season); df['data_year']=df.year
    df.to_csv(os.path.join(BASE,'data','cleaned_air_quality.csv'),index=False)
    return df

def pipe(model):
    return Pipeline([('preprocessor',ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUM),('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),CAT)])),('model',model)])
def score(y,p):
    mse=mean_squared_error(y,p); return {'MAE':mean_absolute_error(y,p),'MSE':mse,'RMSE':np.sqrt(mse),'R2':r2_score(y,p)}

def main():
    df=prep(); d=df[FEATURES+[TARGET]]; cut=int(len(d)*.8); tr,te=d.iloc[:cut],d.iloc[cut:]; Xtr,ytr=tr[FEATURES],tr[TARGET]; Xte,yte=te[FEATURES],te[TARGET]
    models={'Linear Regression':LinearRegression(),'Decision Tree':DecisionTreeRegressor(random_state=42,max_depth=20,min_samples_leaf=2),'Random Forest':RandomForestRegressor(random_state=42,n_estimators=100,max_depth=20,min_samples_leaf=2,n_jobs=-1),'Gradient Boosting':GradientBoostingRegressor(random_state=42,n_estimators=100,max_depth=3,learning_rate=.05)}
    fitted={}; rows=[]
    for name,m in models.items():
        p=pipe(m); p.fit(Xtr,ytr); s=score(yte,p.predict(Xte)); s['Model']=name; rows.append(s); fitted[name]=p
    rf=pipe(RandomForestRegressor(random_state=42,n_jobs=-1)); grid=GridSearchCV(rf,{'model__n_estimators':[50,100],'model__max_depth':[10,15]},cv=2,scoring='neg_root_mean_squared_error',n_jobs=-1); grid.fit(Xtr,ytr); tuned=score(yte,grid.best_estimator_.predict(Xte)); before=next(x for x in rows if x['Model']=='Random Forest')
    if tuned['RMSE']<before['RMSE']: [r.update(tuned) for r in rows if r['Model']=='Random Forest']; fitted['Random Forest']=grid.best_estimator_
    res=pd.DataFrame(rows)[['Model','MAE','MSE','RMSE','R2']].sort_values(['RMSE','R2'],ascending=[True,False]).reset_index(drop=True); best=res.iloc[0]['Model']; bestpipe=fitted[best]
    joblib.dump(bestpipe,os.path.join(MOD,'best_model.pkl')); joblib.dump(bestpipe.named_steps['preprocessor'],os.path.join(MOD,'preprocessor.pkl')); res.to_csv(os.path.join(OUT,'model_comparison.csv'),index=False)
    with open(os.path.join(MOD,'model_metadata.json'),'w') as f: json.dump({'best_model':best,'features':FEATURES,'metrics':res.set_index('Model').to_dict(orient='index'),'train_records':len(tr),'test_records':len(te),'total_records':len(d),'rf_best_params':grid.best_params_},f,indent=2)
    pd.DataFrame({'Actual':yte,'Predicted':bestpipe.predict(Xte)}).to_csv(os.path.join(OUT,'test_predictions.csv'),index=False)
    print(res.to_string(index=False)); print('BEST MODEL:',best)
if __name__=='__main__': main()
