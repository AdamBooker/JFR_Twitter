import json
import pytz
import pandas as pd
from dateutil import parser
from collections import defaultdict
from matplotlib import pyplot as plt

# Address of json file with $ORCL posts
orcl_json = 'in/twtr2015orcl.json'

# Used to convert time zone from GMT to Eastern 
tz = pytz.timezone('US/Eastern') 

# Used to count links to sec.gov for ORCL
orcl_ct = defaultdict(set)
orcl_ct2 = defaultdict(lambda: 0)

with open(orcl_json,encoding='utf8') as f:
    for i in f:
        j = json.loads(i)
        date_ztime = parser.parse(j['created_at'])
        # translate GMT (also called Z-time) to date with EST timezone reference to connect to US market data
        date_est = date_ztime.astimezone(tz).date()

        # Use this to get the number of users posting
        orcl_ct[date_est].add(j['user']['id_str'])
        
        # Use this to get the number of posts
        orcl_ct2[date_est] += 1

orcl_numusers = {}
for i in orcl_ct: orcl_numusers[i] = len(orcl_ct[i])

# Make a DataFrame with the number of users for each day using Twitter data
df_twtr = pd.DataFrame(data=orcl_numusers.items(),columns=['date','numusers'])

#convert from date to datetime for merge
df_twtr['date'] = pd.to_datetime(df_twtr['date'])

# Make a DataFrame with the number of posts for each day using Twitter data
# df_twtr = pd.DataFrame(data=orcl_ct2.items(),columns=['date','numposts'])

edgar_file_loc = 'in/edgar2015.csv'

# Read file with counts of algorithmic and non-algortihmic EDGAR veiws for the firm associated with $ORCL stock 
df_edgar = pd.read_csv(edgar_file_loc)
df_edgar['date'] = pd.to_datetime(df_edgar['date'], format='%m/%d/%Y')

# Merge the two files
df = pd.merge(df_edgar,df_twtr, how='left', on = 'date')

# Make a plot of EDGAR views and Twitter attention 
fig, ax1 = plt.subplots(figsize=(15,3))
ax1.plot(df['date'], df['noalgo_ct'], color = 'green', label='EDGAR')
ax1.set_ylabel('# EDGAR IP Views')
ax2 = ax1.twinx()
ax2.plot(df['date'], df['numusers'], color='blue', label='Twitter')
ax2.set_ylabel('# Twitter Users')
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc=2)
plt.title('Non-algorithmic EDGAR Views and Twitter Attention')


