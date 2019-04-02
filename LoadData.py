import time
import datetime
import pandas as pd
import numpy as np
import xmlrpc.client
import urllib.request,urllib.parse
from interval import Interval
from epics import ca

#get TXT file data
def getTXTpv(filepath,filename):
    re=pd.read_table(filepath+'\\'+filename,header=None)
    return re[0].values.tolist()

#get live date with pvnames
def generate_live_data(duration,period, pvnames):
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
        time.sleep(period)
        duration = duration - period
    ca.finalize_libca(maxtime=1.0)
    df = pd.DataFrame(alldata, columns=cols)
    return df

#Load Channel Archiver data
def connectChanArch(ipaddr):
    sp = '%s%s%s' % ('http://', ipaddr, '/cgi-bin/archiver/ArchiveDataServer.cgi')
    server = xmlrpc.client.ServerProxy(sp)
    engine = server.archiver.archives()
    return server,engine

#using engine name to find key
def getChanArchEngineKey(ipaddr,enginename):
    server,engine=connectChanArch(ipaddr)
    for e in engine:
        if e.get('name')==enginename:
            return e.get('key')

def getChanArchEngineName(ipaddr,enginekey):
    server, engine = connectChanArch(ipaddr)
    for e in engine:
        if e.get('key')==enginekey:
            return e.get('name')

#get history data from Channel Archiver
#return dict with pv name and its data
def getChanArch(ipaddr, key, pvnames, start, end, how=0):
    data = {}
    server, engine = connectChanArch(ipaddr)
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
    server, engine = connectChanArch(ipaddr)
    namelist=[]
    for e in engine:
        try:
            namelist=server.archiver.names(e['key'],'')
        except xmlrpc.client.Fault or xmlrpc.client.ProtocolError as err:
            print("A fault occurred")
            print("Fault string: %s" % err.faultString)
        name = {}
        for nl in namelist:
            name[nl['name']]=e['key']
        keypvlist[e['name']]=name
    for pv in pvnames:
        flag = 0
        for key,value in keypvlist.items():
            if pv in value:
                flag=1
                print(pv,":engine name is: ",key,",engine key is:",value.get(pv))
        if flag==0:
            print(pv," not found.")

def getKeyWithTime(server, engine,pvnames,start,end):
    keypvlist = {}
    namelist = []
    keylists={}
    rekey=None
    for e in engine:
        try:
            namelist = server.archiver.names(e['key'], '')
        except xmlrpc.client.Fault or xmlrpc.client.ProtocolError as err:
            print("A fault occurred")
            print("Fault string: %s" % err.faultString)
        name = {}
        for nl in namelist:
            name[nl['name']] = e['key']
        keypvlist[e['name']] = name
    for pv in pvnames:
        flag = 0
        for key, value in keypvlist.items():
            if pv in value:
                enginestart=datetime2utc(key.split(':')[1].split('-')[0],'%y%m%d')
                if key.split(':')[1].split('-')[1]=='now':
                    engineend=int(time.time())
                else:
                    engineend=int(datetime2utc(key.split(':')[1].split('-')[1], '%y%m%d'))
                enginezoom=Interval(enginestart,engineend)
                if (int(datetime2utc(start)) in enginezoom) and (int(datetime2utc(end)) in enginezoom):
                    flag=1
                    print(pv, ":engine name is: ", key, ",engine key is:", value.get(pv))
                    if value.get(pv) in keylists:
                        keylists[value.get(pv)].append(pv)
                    else:
                        keylists[value.get(pv)]=[]
                        keylists[value.get(pv)].append(pv)
                elif (int(datetime2utc(start)) in enginezoom) ^ (int(datetime2utc(end)) in enginezoom):
                    flag=1
                    print('Time period too big,more than mone engine,',pv, ":engine name is: ", key, ",engine key is:", value.get(pv)," Need to be in one engine to export.")
        rekey = keylists
        if flag == 0:
            print(pv, " not found.")
            rekey=None
    return rekey



#get formatted history data from Channel Archiver
#return format dataFrame
#ipaddr:server ip address
# key:enginekey
#pvnames:list
# startsec,startnano,endsec,endnano
#merge_type=outer:use smallest period data time, fill others
#merge_type=inner:use biggest period data time, delete others
#merge_type=number:user defined time period and merge
#interpolate_type：{'linear','time','index','values', 'nearest','zero','slinear', 'quadratic','cubic','barycentric','krogh','polynomial', 'spline', 'piecewise_polynomial', 'from_derivatives', 'pchip', 'akima'},default linear
#fillna_type:{'backfill','pad', None}, default None
# how:0-raw,1-spreadsheet,2-avg,3-plot-binning,4-linear
# count:sample numbers,how==0-raw data number;others-get data with 1Hz
def getFormatChanArch(server,engine, pvnames, start, end, merge_type='inner', interpolate_type='linear',
                          fillna_type=None, how=0, dropna=True):
        df = pd.DataFrame()
        result = pd.DataFrame()
        dl=[]
        keylists = getKeyWithTime(server, engine, pvnames, start, end)
        if keylists=={}:
            print('Please change time period.')
            return None
        if how == 0:
            count =4 * (int(datetime2utc(end)) - int(datetime2utc(start)))
        else:
            count = int(datetime2utc(end)) - int(datetime2utc(start))
        if merge_type.isdigit():
            timeSeries = pd.date_range(start=start, end=end, freq=merge_type + 'S')
            merge_type = 'left'
            df = pd.DataFrame(timeSeries, columns=['time'])
        for key in keylists:
            datalist={}
            datalen=10000
            newstart=int(datetime2utc(start))
            while datalen==10000:
                datalist = server.archiver.values(key, keylists[key], newstart, 0, int(datetime2utc(end)), 0, count,how)
                datalen=len(datalist[0].get('values'))
                #print('len:',datalen)
                #print(datalist[0].get('values')[datalen-1].get('secs'))
                newstart = datalist[0].get('values')[datalen-1].get('secs')
                dl.append(datalist)
            for d in dl:
                for l in d:
                    timelist = []
                    valuelist = []
                    if len(l.get('values')) == 1:
                        gettimestr = str(pd.to_datetime(l.get('values')[0].get('secs'), unit='s') + datetime.timedelta(hours=8))
                        gettime = time.mktime(time.strptime(gettimestr, '%Y-%m-%d %H:%M:%S'))
                        print(gettime)
                        starttime = time.mktime(time.strptime(start, '%Y/%m/%d %H:%M:%S'))
                        endtime = time.mktime(time.strptime(end, '%Y/%m/%d %H:%M:%S'))
                        if starttime > gettime or endtime < gettime:
                            timelist = list(pd.date_range(start=start, end=end, freq="1" + 'S'))
                            for i in range(len(timelist)):
                                valuelist.append(l.get('values')[0].get('value')[0])
                    for d in l.get('values'):
                        timelist.append(pd.to_datetime(d.get('secs'), unit='s') + datetime.timedelta(hours=8))
                        valuelist.append(d.get('value')[0])
                    if df.empty:
                        df = pd.DataFrame({'time': timelist, l.get('name'): valuelist}).drop_duplicates('time', keep='first')
                    else:
                        df = pd.merge(df, pd.DataFrame({'time': timelist, l.get('name'): valuelist}).drop_duplicates('time',keep='first'),how=merge_type)
                result=pd.concat([result,df])
        if (fillna_type != None):
            result = result.set_index(['time']).sort_index(ascending=True).fillna(method=fillna_type)
        else:
            result = result.set_index(['time']).sort_index(ascending=True).interpolate(method=interpolate_type)
        if dropna == True:
            return result.dropna(axis=0)
        else:
            return result

#get history data from Archiver Appliance
def getArchAppl(data_retrieval_url,pvnames,start,end,merge_type):
    df=pd.DataFrame()
    if merge_type.isdigit():
        timeSeries = pd.date_range(start=start, end=end, freq=merge_type + 'S')
        merge_type = 'left'
        df = pd.DataFrame(timeSeries, columns=['time'])

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
    filetype= filename.split('.')[1] if ('.' in filename) else ''
    if filetype.lower()=='csv':
        try:
            data = pd.read_csv(filepath+'\\\\'+filename, encoding='gb2312', skiprows=skiprows).dropna()
            return data
        except FileNotFoundError:
            print("FileNotFoundError")
    else:
        try:
            data = pd.read_table(filepath + '\\\\' + filename, encoding='gb2312', skiprows=skiprows).dropna()
            return data
        except FileNotFoundError:
            print("FileNotFoundError")

def dataset2df(dataset):
    column = dataset.feature_names
    data = dataset.data
    target = dataset.target
    datadf = pd.DataFrame(data, columns=column)
    datadf['target'] = target
    return datadf

def np2df(data,col=""):
    if isinstance(data,np.ndarray):
        print("ndarray")
        if isinstance(col,list) and len(col)== data.ndim:
            df=pd.DataFrame(data,columns=col)
            return df
        else:
            print("wrong column names,need to be list and same length of data feature nums")
            return
    elif isinstance(data,list):
        print("list")
        if isinstance(col, list) and len(col) ==len(data[0]):
            df = pd.DataFrame(data, columns=col)
            return df
        else:
            print("wrong column names,need to be list and same length of data feature nums")
            return
    elif isinstance(data,dict):
        print("dict")
        print(data)
        df=pd.DataFrame(data)
        return df

def df2np(data,datatype):
    if datatype=="list":
        return data.values.tolist()
    elif datatype=='nparray':
        return np.array(data.values.tolist())
    else:
        return data.to_dict('dict')

def df2other(data,type,path_or_buf=None,encoding='utf-8'):
    if type=='csv':
        print("dasda")
        return data.to_csv(path_or_buf=path_or_buf, encoding=encoding)
    elif type=='html':
        return data.to_html(buf=path_or_buf)
    elif type=='json':
        return data.to_json(path_or_buf=path_or_buf)
    elif type=='excel':
        return data.to_excel(excel_writer=path_or_buf, encoding=encoding)
    elif type=='clipboard':
        return data.to_clipboard()
    else:
        print('Wrong output type! Options:csv,excel,json,html,clipboard.')
        return None

#from format datetime to unix time
def datetime2utc(datestr,dtformat='%Y/%m/%d %H:%M:%S'):
    timestamp = time.mktime(datetime.datetime.strptime(datestr, dtformat).timetuple())
    return timestamp

# keep left dataFrame data
#merge_type:outer inner left right
def mergeDF(left,right,merge_type='left'):
    data=left.join(right,how=merge_type).dropna()
    return data

def compare_time(time1,time2):
    s_time = time.mktime(time.strptime(time1,'%Y/%m/%d %H:%M:%S'))
    e_time = time.mktime(time.strptime(time2,'%Y/%m/%d %H:%M:%S'))
    print (s_time ,'is:',s_time)
    print (e_time ,'is:',e_time)
    return int(s_time) - int(e_time)