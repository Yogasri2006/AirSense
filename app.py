import os,json,joblib,numpy as np,pandas as pd
from flask import Flask,render_template,request,jsonify
BASE=os.path.dirname(os.path.abspath(__file__)); DP=os.path.join(BASE,'data','cleaned_air_quality.csv'); MP=os.path.join(BASE,'model','best_model.pkl'); META=os.path.join(BASE,'model','model_metadata.json'); CP=os.path.join(BASE,'outputs','model_comparison.csv')
app=Flask(__name__); FEATURES=['PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM','wd','month','hour']; model=None; meta={}; df=pd.DataFrame()
if os.path.exists(MP): model=joblib.load(MP)
if os.path.exists(META): meta=json.load(open(META,encoding='utf8'))
if os.path.exists(DP): df=pd.read_csv(DP)
def level(x): return ('Good','Low PM2.5 level.') if x<=35 else ('Moderate','Moderate PM2.5 level.') if x<=75 else ('Unhealthy','Elevated PM2.5 level; continue monitoring.') if x<=150 else ('Very Unhealthy','Very high PM2.5 level.') if x<=250 else ('Hazardous','Extremely high PM2.5 level.')
def val(x,n):
 try:
  x=float(x); assert np.isfinite(x); return x
 except: raise ValueError(f'Enter a valid value for {n}.')
@app.route('/')
def home(): return render_template('index.html',best=meta.get('best_model','Train model first'))
@app.route('/dashboard')
def dashboard():
 if df.empty:return render_template('dashboard.html',error='Run python train_model.py first.')
 return render_template('dashboard.html',stats={'avg':round(df['PM2.5'].mean(),2),'max':round(df['PM2.5'].max(),2),'min':round(df['PM2.5'].min(),2),'records':len(df)},month=df.groupby('month')['PM2.5'].mean().round(2).to_dict(),year=df.groupby('year')['PM2.5'].mean().round(2).to_dict(),season=df.groupby('season')['PM2.5'].mean().round(2).to_dict(),hour=df.groupby('hour')['PM2.5'].mean().round(2).to_dict(),best=meta.get('best_model','N/A'))
@app.route('/prediction',methods=['GET','POST'])
def prediction():
 result=None; error=None
 if request.method=='POST':
  try:
   d={c:val(request.form.get(c),c) for c in FEATURES if c!='wd'}; d['wd']=(request.form.get('wd') or 'Unknown').strip(); x=pd.DataFrame([d],columns=FEATURES); y=float(model.predict(x)[0]); lv,msg=level(y); result={'value':round(y,2),'level':lv,'msg':msg,'model':meta.get('best_model','Best model')}
  except Exception as e:error=str(e)
 return render_template('prediction.html',result=result,error=error)
@app.route('/predict',methods=['POST'])
def api():
 try:
  d=request.get_json() or request.form; x={c:(d.get(c) if c=='wd' else val(d.get(c),c)) for c in FEATURES}; y=float(model.predict(pd.DataFrame([x],columns=FEATURES))[0]); lv,msg=level(y); return jsonify(predicted_pm25=round(y,2),pollution_level=lv,interpretation=msg,model=meta.get('best_model'))
 except Exception as e:return jsonify(error=str(e)),400
@app.route('/models')
def models():
 r=pd.read_csv(CP).round(4).to_dict('records') if os.path.exists(CP) else []; return render_template('models.html',rows=r,best=meta.get('best_model','N/A'))
@app.route('/analysis')
def analysis():
 if df.empty:return render_template('analysis.html',error='Run train_model.py first.',corr=[])
 c=df.select_dtypes('number').corr()['PM2.5'].drop('PM2.5').sort_values(key=abs,ascending=False); return render_template('analysis.html',corr=[(k,round(float(v),3)) for k,v in c.items()])
@app.route('/dataset')
def dataset():
 if df.empty:return render_template('dataset.html',error='Run train_model.py first.')
 x=df.head(10).fillna('').astype(str); return render_template('dataset.html',cols=x.columns.tolist(),rows=x.values.tolist(),records=len(df),features=len(df.columns),station=df.station.iloc[0] if 'station' in df else 'Aotizhongxin')
@app.route('/about')
def about():return render_template('about.html')
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
