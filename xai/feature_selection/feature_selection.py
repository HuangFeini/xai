import pandas as pd
from sklearn import ensemble
import datetime as dt



def time_add(data):
    '''
    Add time-relating features to input matrix.
    DAY: day sequence of entire observation
    DOY: day of year

    :parameter
    file: data ,  from csv
    :returns
    data:pd.Dataframe

    '''
    DAY=[i for i in range(1,data.shape[0]+1)]
    DOYList=[]

    for i in data['TIMESTAMP']:
        time=str(i)
        timet=dt.datetime.strptime(time, "%Y%m%d")
        doy=timet.timetuple().tm_yday
        DOYList.append(doy)


    data['DAY']=DAY
    data['DOY']=DOYList
    data=data.drop(['TIMESTAMP'], axis=1)

    return data

def lag_add(data,sm_lag=7,p_lag=7):
    if sm_lag-0>0:
        sm=data["SWC_F_MDS_1"].tolist()
        lag_col=["SM"+str(x) for x in range(1,sm_lag+1)]

        lag_frame=pd.DataFrame(columns=lag_col)
        i=0
        for col in lag_col:
            i=i+1
            lag=[int(-9999)]*i+sm[:-i]
            lag_frame[col]=lag
        data=pd.concat([data,lag_frame],axis=1)

    if p_lag-0>0:
        p = data["P_F"].tolist()
        lag_col = ["P" + str(x) for x in range(1, p_lag + 1)]

        lag_frame = pd.DataFrame(columns=lag_col)
        i = 0
        for col in lag_col:
            i = i + 1
            lag = [int(-9999)] * i + p[:-i]
            lag_frame[col] = lag
        data = pd.concat([data, lag_frame], axis=1)

        return data
    else:
        return data









class split_data():
    '''
    Split dataset to training set and testing set.
    According to the time-sequence (using data of time-ahead to predict feature data).
    ------
    :parameter
    data[dataframe]: A data set
    part[float]: division proportion for two default as 0.7
    part3[list]: A list division proportion for three, default as [0.7,0.2,0.1]

    :returns
    dataset of train, test
    '''
    def __init__(self,data,part=0.7,part3=[0.7,0.2,0.1]):
        self.data=data
        self.part=part
        self.part3=part3

    def split(self):
        length=self.data.shape[0]
        split=round(length*self.part)
        train=self.data[0:split]
        test=self.data[split:length]
        return train,test

    def split3(self):
        length=self.data.shape[0]
        split1=round(length*self.part3[0])
        train=self.data[0:split1]

        split2=round(length*self.part3[1]+split1)
        valid=self.data[split1:split2]

        test=self.data[split2:length]
        return train, valid, test







class data_processing():
    '''
    Data washing.
    :parameter
    data: raw data, dataframe(after adding time-relating features)
    :returns
    Newdata: dataframe
    '''
    def __init__(self,data):
        self.data=data

    def elim_SM_nan(self):
        '''
        Eliminate the observation without SM values.
        '''
        length = self.data.shape[0]
        SM = self.data["SWC_F_MDS_1"].values.tolist()
        nanNum = SM.count(-9999)
        nanLimit = 0.3 * length
        if nanNum > nanLimit:
            print("NaN in SM exceed the limit of 30%, please choose another site.")
            return None
        else:
            NewData=self.data[-self.data["SWC_F_MDS_1"].isin([-9999])]
            return NewData

    def drop_ir(self):
        '''
        Eliminate irrelevant records in FLUXNET,
        like percentiles, quality index, RANDUNC, se, sd...
        '''
        EZList=['JOINTUNC','QC','SE','SD', 'RANDUNC','_05','_16','_25','_75','_50','_95','_84']
        FeaList = self.data.columns.tolist()
        DropFeaList=[]
        for i in FeaList:
            for j in EZList:
                if j in i:
                    DropFeaList.append(i)

        NewData=self.data.drop(DropFeaList,axis=1)

        return NewData

    def drop_nan_feature(self):
        '''
        Eliminate the features with too many(30%) Nan.
        '''
        length = self.data.shape[0]
        nanLimit = 0.3 * length
        DropNanList=[]

        for f in self.data.columns.tolist():
            value=self.data[f].values.tolist()
            nanNum = value.count(-9999)
            if nanNum > nanLimit:
                DropNanList.append(f)

        NewData = self.data.drop(DropNanList, axis=1)

        return NewData


class feature_selection():
    '''
    Using sequential backward selection (SBS) to select optimal features.
    Based on random forest (RF).
    :parameter
    data: dataframe
    :returns
    NFeaFrame：dataframe, including features and their importance calculated by MSE.

    '''
    def __init__(self,data):
        self.data=data

    def sbs_rf(self,n_estimators):


        FeaList=self.data.columns.tolist()
        FeaList.remove("SWC_F_MDS_1")
        NFeaFrame=pd.DataFrame(columns=['ElimFeature','score'])
        x = self.data.drop("SWC_F_MDS_1", axis=1)
        for i in range(1,self.data.shape[1]):
            RF=ensemble.RandomForestRegressor(n_estimators=n_estimators)
            y=self.data["SWC_F_MDS_1"]
            RF.fit(x,y)
            score=RF.score(x,y)

            fi=RF.feature_importances_
            fii={'Feature':x.columns.tolist(), 'Importance':fi}
            fiFrame=pd.DataFrame(fii)


            minFea=fiFrame['Feature'][fiFrame['Importance']==fiFrame['Importance'].min()]

            xm=minFea.tolist()[0]
            NFeaFrame.loc[i]=[xm,score]
            x=x.drop(minFea, axis=1)



        return NFeaFrame

class data_processing_main():
    def __init__(self, data,time_add,lag_add,elim_SM_nan, drop_ir, drop_nan_feature,part,n_estimator,sbs):
        self.data=data
        self.time_add=time_add
        self.lag_add=lag_add
        self.elim_SM_nan=elim_SM_nan
        self.drop_ir=drop_ir
        self.drop_nan_feature=drop_nan_feature
        self.part=part
        self.n_estimator=n_estimator
        self.sbs=sbs





    def total(self):


        if self.time_add:
            self.data = time_add(self.data)

        if self.lag_add:
            self.data=lag_add(self.data)



        if self.elim_SM_nan:
            fa1=data_processing(self.data)
            self.data=fa1.elim_SM_nan()
        if self.drop_ir:
            fa2 = data_processing(self.data)
            self.data=fa2.drop_ir()
        if self.drop_nan_feature:
            fa3=data_processing(self.data)
            self.data=fa3.drop_nan_feature()
        if self.sbs:
            sd=split_data(self.data,part=self.part)
            train,test=sd.split()
            fb=feature_selection(train)

            feature_sequence=fb.sbs_rf(n_estimators=self.n_estimator)

            return self.data,feature_sequence
        else:
            return self.data




if __name__ == '__main__':

    file='D:\\codes\\xai\\flx_data\\FLX_CN-Ha2_FLUXNET2015_FULLSET_DD_2003-2005_1-4.csv'
    data=pd.read_csv(file,header=0)

    d=data_processing_main(data=data,
                          time_add=1,
                           lag_add=1,
                          elim_SM_nan=1,
                          drop_ir=1,
                          drop_nan_feature=1,
                          part=0.7,
                          n_estimator=10,
                          sbs=True)
    dd,ss=d.total()
    dd.to_csv("dd.csv")
    ss.to_csv('ss.csv')




















