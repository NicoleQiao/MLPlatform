import time
import datetime
import pandas as pd
import numpy as np
import xmlrpc.client
import urllib.request,urllib.parse
import json
from epics import ca

#get live date with pvnames
def generate_live_data(duration, pvnames):
    pvnamelist = tuple(pvnames)
    cols = list(pvnamelist)
    cols.insert(0, 'timestamp')
    pvconn = {}
    alldata = []
    ca.initialize_libca()
    for name in pvnamelist:
        chid = ca.create_channel(name, connect=False, auto_cb=False)
        pvconn[name] = (chid, None)
    for key in list(pvconn):
        state = ca.connect_channel(pvconn.get(key)[0])
        if not state:
            print('PV:', key, 'not connected')
            pvconn.pop(key)
    ca.poll()
    while duration > 0 and pvconn:
        pvdata = []
        for name, data in pvconn.items():
            ca.get(data[0], wait=False)
        ca.poll()
        pvdata.append(datetime.datetime.now())
        for name, data in pvconn.items():
            val = ca.get_complete(data[0])
            pvdata.append(val)
        alldata.append(pvdata)
        time.sleep(1)
        duration = duration - 1
    ca.finalize_libca(maxtime=1.0)
    df = pd.DataFrame(alldata, columns=cols)
    return df

#using engine name to find key
def getChanArchAllEngineKey(ipaddr,enginename):
    sp = '%s%s%s' % ('http://', ipaddr, '/cgi-bin/archiver/ArchiveDataServer.cgi')
    server = xmlrpc.client.ServerProxy(sp)
    engine = server.archiver.archives()
    for e in engine:
        if e.get('name')==enginename:
            return e.get('key')
    return 0


#from format datetime to unix time
def datetime2utc(datestr):
    dtformat = '%m/%d/%Y %H:%M:%S'
    timestamp = time.mktime(datetime.datetime.strptime(datestr, dtformat).timetuple())
    return timestamp

#get history data from Channel Archiver
#return dict with pv name and its data
def getChanArch(ipaddr, key, pvnames, start, end, how=0):
    data = {}
    sp = '%s%s%s' % ('http://', ipaddr, '/cgi-bin/archiver/ArchiveDataServer.cgi')
    server = xmlrpc.client.ServerProxy(sp)
    #key:engine index;pvnames:list;startsec,startnano,endsec,endnano,count:sample numbers;how:0-raw,1-spreadsheet,2-avg,3-plot-binning,4-linear
    if how==0:
        count=4*(int(datetime2utc(end))-int(datetime2utc(start)))
    else:
        count=int(datetime2utc(end))-int(datetime2utc(start))
    datalist = server.archiver.values(key, pvnames, int(datetime2utc(start)), 0, int(datetime2utc(end)), 0, count,how)
    for l in datalist:
        timelist = []
        valuelist = []
        for d in l.get('values')[1:]:
            timelist.append(d.get('secs'))
            valuelist.append(d.get('value')[0])
        data[l.get('name')] = pd.DataFrame({'time': timelist, l.get('name'): valuelist}).drop_duplicates('time',keep='first')
    return data


def getKey(ipaddr,pvnames):
    keypvlist={}
    sp = '%s%s%s' % ('http://', ipaddr, '/cgi-bin/archiver/ArchiveDataServer.cgi')
    server = xmlrpc.client.ServerProxy(sp)
    engine = server.archiver.archives()
    #print(engine)
    for e in engine:
        namelist=server.archiver.names(e['key'],'')
        name = {}
        for nl in namelist:
            name[nl['name']]=e['key']
        keypvlist[e['name']]=name
    for pv in pvnames:
        for key,value in keypvlist.items():
            if pv in value:
                print(pv,":engine name is: ",key,",engine key is:",value.get(pv))




#get history data from Channel Archiver
#return format dataFrame
#ipaddr:server ip address
# key:enginekey
#pvnames:list
# startsec,startnano,endsec,endnano
#merge_type=outer:use smallest period data time, fill others
#merge_type=inner:use biggest period data time, delete others
#merge_type=number:user defined time period and merge
#interpolate_type：{'linear','time','index','values', 'nearest','zero','slinear', 'quadratic','cubic','barycentric','krogh','polynomial', 'spline', 'piecewise_polynomial', 'from_derivatives', 'pchip', 'akima'},default linear
#fillna_type:{'backfill', 'bfill', 'pad', 'ffill', None}, default None
# how:0-raw,1-spreadsheet,2-avg,3-plot-binning,4-linear
# count:sample numbers,how==0-raw data number;others-get data with 1Hz
def getFormatChanArch(ipaddr, key, pvnames, start, end, merge_type,interpolate_type='linear',fillna_type=None,how=0,dropna=True):
    df=pd.DataFrame()
    sp = '%s%s%s' % ('http://', ipaddr, '/cgi-bin/archiver/ArchiveDataServer.cgi')
    server = xmlrpc.client.ServerProxy(sp)
    namelist = {}
    names = server.archiver.names(key, '')
    for name in names:
        namelist[name['name']]=1
    #print(namelist)
    for pv in pvnames:
        if pv not in namelist:
            print(pv,"is not in this engine,please use getKey(ipaddr,pvnames) to find the right engine")
            pvnames.remove(pv)
    if how == 0:
        count = 4 * (int(datetime2utc(end)) - int(datetime2utc(start)))
    else:
        count = int(datetime2utc(end)) - int(datetime2utc(start))
    if merge_type.isdigit():
        timeSeries = pd.date_range(start=start, end=end, freq=merge_type+'S')
        merge_type = 'left'
        df = pd.DataFrame(timeSeries, columns=['time'])
    datalist = server.archiver.values(key, pvnames, int(datetime2utc(start)), 0, int(datetime2utc(end)), 0, count, how)
    for l in datalist:
        timelist = []
        valuelist = []
        for d in l.get('values')[1:]:
            timelist.append(pd.to_datetime(d.get('secs'),unit='s')+datetime.timedelta(hours=8))
            valuelist.append(d.get('value')[0])
        if df.empty:
            df=pd.DataFrame({'time': timelist, l.get('name'): valuelist}).drop_duplicates('time', keep='first')
        else:
            df=pd.merge(df,pd.DataFrame({'time': timelist, l.get('name'): valuelist}).drop_duplicates('time', keep='first'),how=merge_type)
    if(fillna_type!=None):
        df=df.set_index(['time']).sort_index(ascending=True).fillna(method=fillna_type)
    else:
        df=df.set_index(['time']).sort_index(ascending=True).interpolate(method=interpolate_type)
    if dropna==True:
        return df.dropna(axis = 0)
    else:
        return  df
#get history data from Archiver Appliance
def getArchAppl(data_retrieval_url,pvnames,start,end,merge_type):
    df=pd.DataFrame()
    if merge_type.isdigit():
        timeSeries = pd.date_range(start=start, end=end, freq=merge_type + 'S')
        merge_type = 'left'
        df = pd.DataFrame(timeSeries, columns=['time'])
    data_retrieval_url='192.168.44.168:17665/retrieval'
    urlp='%s%s%s%s'%('http://',data_retrieval_url,'/data/getData.csv','?')
    getp = {}
    getp['pv']=''
    getp['from']=start
    getp['to'] = end
    for pv in pvnames:
        getp['pv']=pv
        s=urllib.parse.urlencode(getp)
        getdata=urllib.request.urlopen(urlp+s)
        df=pd.merge(df,pd.read_csv(getdata),how=merge_type)
    #urllib.request.urlopen("http://192.168.44.168:17665/retrieval/data/getData.json?pv=HEBT_Mag%3AQV05%3AB&from=2018-04-02T14%3A00%3A00.000Z&to=2018-04-02T15%3A00%3A00.000Z")
    return df
#use standard CSV File format
def getLocalFile(filepath,filename,skiprows=0):
    filetype=filename.split('.')[1]
    if filetype.lower()=='csv':
        data = pd.read_csv(filepath+'\\'+filename, encoding='gb2312', skiprows=skiprows).dropna()
    else:
        data = pd.read_table(filepath + '\\' + filename, encoding='gb2312', skiprows=skiprows).dropna()
    print(data)
    return data
# keep left dataFrame data
#merge_type:outer inner left right
def mergeDF(left,right,merge_type):
    data=left.join(right,how=merge_type).dropna()
    return data
def getTXTpv(filepath,filename):
    re=pd.read_table(filepath+'\\'+filename,header=None)
    #print(re[0].values.tolist())
    return re[0].values.tolist()

def np2df(dataset):
    column = dataset.feature_names
    data = dataset.data
    target = dataset.target
    df1 = pd.DataFrame(data, columns=column)
    df2 = pd.DataFrame(target, columns=['target'])
    data_df = pd.concat([df1, df2], axis=1)
    return data_df