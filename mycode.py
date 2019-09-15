import eikon as ek # Eikon
import numpy as np # Numpy
import pandas as pd # Pandas

#import cufflinks as cf # Cufflinks
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib as mpl
import os # To get current director etc

from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# Useful references
# <http://strftime.org/>

# Code from
# https://rightfootin.blogspot.com/2006/09/more-on-python-flatten.html

def flatten(l):
    out = []
    for item in l:
        if isinstance(item, (list, tuple)):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out

def ordered_unique(list_):
    unique_items = []
    for x in list_:
        if x not in unique_items: unique_items.append(x)
    return unique_items

def rev_iso_date(ds='yyyy-mm-dd'):
    """ Returns iso date in dd-mm-yyyy format
    """
    
    year = ds[:4]
    month = ds[5:7]
    day = ds[8:]
    
    return day+'-'+month+'-'+year

def suggest_filename_ts(df, ts, desc_string, ext):
    """Returns suggested filename for dataframe containing timeseries
    """
    end = ts.max().isoformat().split('T')[0]
    start = ts.min().isoformat().split('T')[0]
    return desc_string+'_'+start+'_'+end+'.'+ext

def suggest_filename(desc_string, ext):
    """Returns suggested filename for dataframe containing timeseries
    """
    return pd.Timestamp.now().strftime('%Y-%m-%d-%H%M')+'-'+desc_string+'.'+ext

def time_to_chunks(start_date_string, end_date_string, days):
    """Divide the time between two dates into chunks -- useful as the eikon API limits the return to 3000 datapoints

By way of example:
---

list_of_dfs = []
for s,f in time_to_chunks('2019-01-01', '2019-08-07', 30):
    df_t = ek.get_timeseries(list(df_sector_lookup.sec_ric.values),fields="CLOSE", \
                        start_date=s, 
                        end_date=f, interval='daily')
    list_of_dfs.append(df_t.copy(deep=True))

df_combined = pd.concat(list_of_dfs, axis=0).drop_duplicates(keep='first').sort_index()
    """
    periods = 0
    
    start_date = pd.to_datetime(start_date_string)
    end_date = pd.to_datetime(end_date_string)
    date = pd.to_datetime(end_date_string)
    
    # Go back in time and count periods
    while (date > start_date):
        date -= pd.Timedelta(days, unit='days')
        periods += 1
    
    dates = []
    for i in range(0,periods):
        dates.append(
            (
                (end_date - pd.Timedelta(days*(i+1), unit='days')).isoformat().split('T')[0], 
                (end_date - pd.Timedelta(days*i, unit='days')).isoformat().split('T')[0]
            )
        )
        end_date -= pd.Timedelta(1, unit='days')
    
    return dates

def day_delta(date_string, d):
    """Returns date_string as pd_dt with day +/-d
    """
    return pd.to_datetime(date_string) + pd.Timedelta(d, unit='days')

def month_delta(date_string, d):
    """Returns date_string as pd_dt with month +/-d
    """
    return pd.to_datetime(date_string) + pd.Timedelta(d, unit='M')

def year_delta(date_string, d):
    """Returns date_string as pd_dt with year +/-d
    """
    return pd.to_datetime(date_string) + pd.Timedelta(d, unit='y')

def day_delta_s(date_string, d):
    """Returns date_string as str with day +/-d
    """
    r = pd.to_datetime(date_string) + pd.Timedelta(d, unit='days')
    return r.strftime('%Y-%m-%d')

def month_delta_s(date_string, d):
    """Returns date_string as str with month +/-d
    """
    r = pd.to_datetime(date_string) + pd.Timedelta(d, unit='M')
    return r.strftime('%Y-%m-%d')

def year_delta_s(date_string, d):
    """Returns date_string as str with year +/-d
    """
    r = pd.to_datetime(date_string) + pd.Timedelta(d, unit='y')
    return r.strftime('%Y-%m-%d')

def uws_two_registers(df_1, df_2, option=None):
    """First two args are dataframes
Options can be:
'graph'
'similar'
if None returns the combined dataframe

Dataframes must look like this:
---
Instrument                                 object
Investor Full Name                         object
Holdings Pct Of Shares Outstanding Held    float64
Investor Shares Held                       int64
Investor Type Description                  object
Investor Parent Type Description           object
---
"""
    df_combo = pd.concat([df_1, df_2], axis=0, ignore_index=True)
    joint_shareholders = [k for k,v in (df_combo.groupby('Investor Full Name').Instrument.count() > 1).iteritems() if v == True]

    holdings_in_each = [(shareholder,
    df_1.loc[df_1['Investor Full Name'] == shareholder, 'Holdings Pct Of Shares Outstanding Held'].values[0],
    df_2.loc[df_2['Investor Full Name'] == shareholder, 'Holdings Pct Of Shares Outstanding Held'].values[0])
    for shareholder in joint_shareholders]
    
    df_reg = pd.DataFrame(holdings_in_each, columns=['Investor',df_1.iloc[0,0],df_2.iloc[0,0]])

    df_reg['Combined'] = df_reg.iloc[:,1] + df_reg.iloc[:,2]
    df_reg['Avg'] = df_reg['Combined']/2
    
    df_reg[df_reg.columns[1] + ' o/w'] = df_reg.loc[:,df_reg.columns[1]]-df_reg.loc[:,'Avg']
    df_reg[df_reg.columns[2] + ' o/w'] = df_reg.loc[:,df_reg.columns[2]]-df_reg.loc[:,'Avg']

    df_reg_s = df_reg[df_reg.Combined > 0].sort_values(by='Combined', ascending=False).reset_index(drop=True)
    
    if option == 'graph':
        return (df_reg_s[df_reg_s.columns[1:]].cumsum(axis = 0)).plot(title='Cumulative measures')
    
    if option == 'similar':
        return ((df_reg.iloc[:,1:] > 0) * 1).corr().iloc[1,0], df_reg.iloc[:,1:].corr().iloc[1,0]
        
    return df_reg_s

def get_share_reg(df_master, ticker):
    """Returns share register of company with ticker as df from df_master
    
df_master:
---
Instrument                                 object
Investor Full Name                         object
Holdings Pct Of Shares Outstanding Held    float64
Investor Shares Held                       int64
Investor Type Description                  object
Investor Parent Type Description           object
---
"""
    return df_master[df_master.Instrument == ticker].reset_index(drop=True).copy(deep=True)

def get_investments(df_master, investor_string):
    """Returns company holdings of investor with investor_string as df from df_master
    
df_master:
---
Instrument                                 object
Investor Full Name                         object
Holdings Pct Of Shares Outstanding Held    float64
Investor Shares Held                       int64
Investor Type Description                  object
Investor Parent Type Description           object
---
"""
    mask = (df_master['Investor Full Name'] == investor_string)
    mask = mask & (df_master['Holdings Pct Of Shares Outstanding Held'] > 0)
    
    return df_master[mask].sort_values(by='Holdings Pct Of Shares Outstanding Held',ascending=False).reset_index(drop=True).copy(deep=True)

def get_investments_named(df_master, investor_string, df_sec):
    """Returns company holdings of investor with investor_string as df from df_master WITH names added as final column
    
df_master:
---
Instrument                                 object
Investor Full Name                         object
Holdings Pct Of Shares Outstanding Held    float64
Investor Shares Held                       int64
Investor Type Description                  object
Investor Parent Type Description           object
---
"""
    r_df = get_investments(df_master, investor_string)
    names = [df_sec.loc[df_sec.Instrument == ticker,'Company Common Name'].values[0] for ticker in r_df.Instrument.values]
    r_df['Name'] = names
    return r_df

def uws_many_registers(df_list, option=None):
    """First arg is a list of dataframes
Options can be:
'graph'
'similar'
if None returns the combined dataframe

Dataframes must look like this:
---
Instrument                                 object
Investor Full Name                         object
Holdings Pct Of Shares Outstanding Held    float64
Investor Shares Held                       int64
Investor Type Description                  object
Investor Parent Type Description           object
---
"""
    df_combo = pd.concat(df_list, axis=0, ignore_index=True)
    joint_shareholders = [k for k,v in (df_combo.groupby('Investor Full Name').Instrument.count() > 1).iteritems() if v == True]
    #return joint_shareholders

    holdings_in_each = \
    [
        (shareholder,
        [
            flatten(df.loc[df['Investor Full Name'] == shareholder, 'Holdings Pct Of Shares Outstanding Held'].values) \
             for df in df_list
        ])
        for shareholder in joint_shareholders
    ]
    
    df_reg = pd.DataFrame([flatten(x) for x in holdings_in_each], columns=['Investor']+ \
            [df.iloc[0,0] for df in df_list]).fillna(value=0)

    df_reg['Combined'] = df_reg.iloc[:,1:len(df_list)+1].sum(axis=1)
    df_reg['Avg'] = df_reg['Combined'] / len(df_list)
     
    for col in df_reg.columns[1:len(df_list)+1]:
        df_reg[col + ' o/w'] = df_reg.loc[:,col]-df_reg.loc[:,'Avg']
        
    df_reg_s = df_reg[df_reg.Combined > 0].sort_values(by='Combined', ascending=False).reset_index(drop=True)
    
    if option == 'graph':
        return (df_reg_s[df_reg_s.columns[1:]].cumsum(axis = 0)).plot(title='Cumulative measures')
    
    if option == 'similar':
        #return ((df_reg_s.iloc[:,1:] > 0) * 1).corr().iloc[1,0], df_reg_s.iloc[:,1:].corr().iloc[1,0]
        return df_reg_s.iloc[:,1:].corr()
    
    return df_reg_s

def combine_two_reg(df_a,df_b):
    """Returns combination of two registers, joined on the investor, which becomes the index
    
dataframes must contain
---
Investor Full Name
Holdings Pct Of Shares Outstanding Held
"""
    of_interest=['Investor Full Name','Holdings Pct Of Shares Outstanding Held']
    return df_a[of_interest].set_index('Investor Full Name').join(df_b[of_interest].set_index('Investor Full Name'),on='Investor Full Name',lsuffix='_a',rsuffix='_b').fillna(value=0)

def register_similarity(df_combo, binary=False):
    """Returns correlation between holdings

Dataframe df_combo expected to be output from combine_two_reg(df_a,df_b)
"""
    if (binary):
        return np.corrcoef(
            ((df_combo['Holdings Pct Of Shares Outstanding Held_a'] > 0) * 1),
            ((df_combo['Holdings Pct Of Shares Outstanding Held_b'] > 0) * 1)
        )[0,1]
    
    return df_combo.corr().iloc[0,1]

def get_ticker(name, df_sec):
    """
Returns ticker based on finding name string in df_sec
--
df_sec=df_sec
"""
    mask = df_sec['Company Common Name'].str.lower().str.contains(name.lower())
    return df_sec.loc[mask,'Instrument'].values[0]

def prep_tr_codes(code,short_code,periods=5):
    """
Assembles dict of codes and periods for passing to ek.get_data

Example
---

my_codes_dict = prep_tr_codes(
    [
        'TR.RevenueActValue',
        'TR.EBITDAActValue',
        'TR.EBITActValue',
        'TR.PreTaxProfitActValue',
        'TR.EPSActValue',
        'TR.DPSActValue'
    ],
    ['Rev','EBITDA','EBIT','PBT','EPS','DPS']
)

codes_for_api = flatten([my_codes_dict[x] for x in my_codes_dict.keys()])

df = ek.get_data('WPP.L',codes_for_api)[0]
"""
    return {k:[ek.TR_Field(v,params=dict(Period='FY'+str(-x))) for x in range(periods)] for k,v in zip(short_code,code)}

def rev_columns(df):
    """
Returns df with column order reversed    
"""
    return df[list(df.columns[::-1])]

def prep_df_after_ek(df, my_codes_dict):
    """
Used to sort out the returned dataframe from ek.get_data()

See also prep_tr_codes()
"""
    short_codes = list(my_codes_dict.keys())
    long_codes = [
        y for y in [list(x.keys())[0] for x in 
        flatten([my_codes_dict[k] for k in my_codes_dict.keys()])]
    ]
    periods = [
        y[x]['params']['Period'] for x,y in zip(long_codes,
        flatten([my_codes_dict[k] for k in my_codes_dict.keys()]))
    ]
    rep = int(len(long_codes)/len(short_codes))
    short_code_rep = flatten([[short_code]*rep for short_code in short_codes])
    new_column_names = [x+'_'+y for x,y in zip(short_code_rep,periods)]
    df.columns = ['Instrument'] + new_column_names
    return df

def order_columns_foward(df,sep='_'):
    """
Given a df with columns e.g.
    Rev_FY0,Rev_FY-1,Rev_FY-2,Rev_FY-3,Rev_FY-4
will be re-ordered to
    Rev_FY-4,Rev_FY-3,Rev_FY-2,Rev_FY-1,Rev_FY0
"""
    # Get shortcodes
    shortcodes = [x.split(sep)[0] for x in df.columns]
    uniquecodes = ordered_unique(shortcodes)
    idx_groups = []
    for uniquecode in uniquecodes:
        idx_groups.append(np.array(range(len([x for x in shortcodes if x == uniquecode]))))
        
    # idx_groups will now look something like the below
    
    # [[0, 1, 2, 3, 4],
    #  [0, 1, 2, 3, 4],
    #  [0, 1, 2, 3, 4],
    #  [0, 1, 2, 3, 4],
    #  [0, 1, 2, 3, 4],
    #  [0, 1, 2, 3, 4]]

    #return idx_groups[0][-1]
    # Add the last index in the previous group to make them contiguous
    for i in range(1,len(idx_groups)):
            idx_groups[i] += idx_groups[i-1][-1]+1
            
    column_order = []
    for group in idx_groups:
        column_order.append(np.flip(group).tolist())
    
    # column_order will now look something like the below
    
    # [[4, 3, 2, 1, 0],
    #  [9, 8, 7, 6, 5],
    #  [14, 13, 12, 11, 10],
    #  [19, 18, 17, 16, 15],
    #  [24, 23, 22, 21, 20],
    #  [29, 28, 27, 26, 25]]
    
    return df.iloc[:,flatten(column_order)]

def prep_tr_codes(code,short_code,periods=None,forward=False):
    """
Assembles dict of codes and periods for passing to ek.get_data

Use forward = True for FY1,2,3 etc and forward = False for FY0,FY-1,FY-2 etc

Example
---

my_codes_dict = prep_tr_codes(
    [
        'TR.RevenueActValue',
        'TR.EBITDAActValue',
        'TR.EBITActValue',
        'TR.PreTaxProfitActValue',
        'TR.EPSActValue',
        'TR.DPSActValue'
    ],
    ['Rev','EBITDA','EBIT','PBT','EPS','DPS']
)

codes_for_api = flatten([my_codes_dict[x] for x in my_codes_dict.keys()])

df = ek.get_data('WPP.L',codes_for_api)[0]
"""
    if (forward):
        if periods == None: periods = 4
        return {k:[ek.TR_Field(v,params=dict(Period='FY'+str(x))) for x in range(1,periods)] for k,v in zip(short_code,code)}
    
    if periods == None: periods = 5
    return {k:[ek.TR_Field(v,params=dict(Period='FY'+str(-x))) for x in range(periods)] for k,v in zip(short_code,code)}

def get_name(ticker, df_sec):
    """
Returns ticker based on finding name string in df_sec
--
df_sec=df_sec
"""
    mask = df_sec['Instrument'].str.lower().str.contains(ticker.lower())
    return df_sec.loc[mask,'Company Common Name'].values[0]

def my_rics(df_sec):
    
# Ignore the below REITs

#  ('MPO.L', 'Macau Property Opportunities Fund Ltd'),
#  ('DJAN.L', 'Daejan Holdings PLC'),
#  ('EPICE.L', 'Ediston Property Investment Company PLC'),
#  ('NETW.L', 'Network International Holdings PLC'),
#  ('SERE.L', 'Schroder European Real Estate Investment Trust PLC'),
#  ('BCPT.L', 'BMO Commercial Property Trust Ltd'),
#  ('BREI.L', 'BMO Real Estate Investments Ltd'),
#  ('LXIL.L', 'LXi REIT PLC'),
#  ('IHR.L', 'Impact Healthcare REIT PLC'),
#  ('SOHO.L', 'Triple Point Social Housing REIT PLC')

    exclude = \
    ['MPO.L',
     'DJAN.L',
     'EPICE.L',
     'SERE.L',
     'BCPT.L',
     'BREI.L',
     'LXIL.L',
     'IHR.L',
     'SOHO.L',
     'TRS.L' # Tarsus (aquired)
     ]

    return [x for x in list(df_sec.loc[df_sec['Fundamental Template'] == 'Industry','Primary Quote RIC']) if x not in exclude]

def growthiness(x,y,periods):
    r = []
    for x,y in zip(x.values,y.values):
        if isinstance(x,(np.ndarray,list)): x=x[0]
        if isinstance(y,(np.ndarray,list)): y=y[0]
            
        # Both positive: normal CAGR
        if (x>0) and (y>0):
            result = (y / x) ** (1/periods)
            
        # Both negative: flip nom/denom so smaller losses = growth
        elif (x<0) and (y<0):
            result = (y / x) ** -(1/periods)
            
        # From negative to positive: instead plug in doubling
        elif (x<0) and (y>0):
            result = 2
            
        # From positive to negative: instead plug in -1
        elif (x>0) and (y<0):
            result = -1
            
        else: result = 0
        
        r.append(result)

    return r

def calc_metrics(df, short_code, stat='cag'):
    columns_select = [column for column in df.columns if short_code+'_FY' in column]
    
    if stat == 'cag': # CAGR
        earliest_year = df[columns_select].loc[:,columns_select[0]]
        latest_year = df[columns_select].loc[:,columns_select[-1]]
        return (latest_year / earliest_year) ** (1/(len(columns_select)-1)) # CAG
    
    if stat == 'acag': # Adjusted CAGR
        earliest_year = df[columns_select].loc[:,columns_select[0]]
        latest_year = df[columns_select].loc[:,columns_select[-1]]
        periods = (len(columns_select)-1)
        return growthiness(earliest_year,latest_year,periods)
        
    if stat == 'sd': # Standard dev
        return df[columns_select].std(axis=1)
    
    if stat == 'mean': # Mean
        return df[columns_select].mean(axis=1)

    if stat == 'lrtp': # Last period relative to peak
        latest_year = df[columns_select].loc[:,columns_select[-1]]
        return latest_year/df[columns_select].max(axis=1)
    
    if stat == 'trtl': # Trough relative to last period
        latest_year = df[columns_select].loc[:,columns_select[-1]]
        return df[columns_select].min(axis=1)/latest_year
    
    if stat == 'gyp': # Grow to shrink ratio
        growth_periods_n = ((df[columns_select].diff(axis=1) > 0) * 1).sum(axis=1)
        return growth_periods_n / (len(columns_select)-1)
    
    if stat == 'ppp': # Postive periods proportion
        calc = ((df[columns_select] > 0) * 1)
        return calc.sum(axis=1) / calc.count(axis=1)
    
    if stat == 'plot':
        df[columns_select].plot(kind='bar')

def get_co_is(df,ticker,extras=False,hist_years=5,est_years=3):
    """
Returns a kind of mini income statement
Expects df to be in the from of df_hist_is
extras = True will include growth rates, margins
Note that growth rates calculate from right to left so year order espected to be ... FY-1,FY0,FY1 ... etc
Define number of historic and forecast years with hist_years and est_years respectively
"""

    idx=list(np.flip(np.array(['FY'+str(-x) for x in range(hist_years)])))
    if (est_years > 0): idx += ['FY'+str(x) for x in range(1,est_years+1)]

    df_out = pd.DataFrame(dict(
    Rev=df.loc[ticker,[c for c in df.columns if 'Rev' in c]].values / 10**6,
    EBITDA=df.loc[ticker,[c for c in df.columns if 'EBITDA_' in c]].values / 10**6,
    EBIT=df.loc[ticker,[c for c in df.columns if 'EBIT_' in c]].values / 10**6,
    PBT=df.loc[ticker,[c for c in df.columns if 'PBT' in c]].values / 10**6,
    EPS=df.loc[ticker,[c for c in df.columns if 'EPS' in c]].values,
    DPS=df.loc[ticker,[c for c in df.columns if 'DPS' in c]].values
            ),index=idx)
    
    if (extras):
        for c in df_out.columns:
                df_out[c+' growth'] = df_out[c].pct_change()

        for c in ['EBITDA','EBIT','PBT']:
            df_out[c+' margin'] = df_out[c] / df_out['Rev']

        # Reorder

        order = [
            'Rev',
            'Rev growth',
            'EBITDA', 
            'EBITDA growth',
            'EBITDA margin',
            'EBIT', 
            'EBIT growth',
            'EBIT margin',
            'PBT', 
            'PBT growth',
            'PBT margin',
            'EPS', 
            'EPS growth',
            'DPS', 
            'DPS growth'
        ]

        df_out = df_out[order]

    return df_out.T

#---

my_dfs = lambda: [x for x in globals() if 'df_' in x]

my_globs = lambda: [x for x in globals() if x[0] != '_' and x[:2] != 'df']

td = lambda: pd.Timestamp.today().strftime('%Y-%m-%d')

#---

def binmap(x):
    """
Maps: 
    nan or 0 -> 0
    >1 -> 1
"""
    if (np.isnan(x)):
        return 0
    if x>0:
        return 1
    return 0

def sparseify(df):
    """
Returns Dataframe with sparse columns
"""
    return pd.DataFrame({k:pd.SparseArray(v) for k,v in zip(df.columns,df.T.values)},index=df.index)

def densify(df_s):
    """
Returns Dataframe with dense columns
"""
    return pd.DataFrame({k:v for k,v in zip(df_s.columns,df_s.T.values)},index=df_s.index)
