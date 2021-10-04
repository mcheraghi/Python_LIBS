
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.parser import parse


print('------------------------------------------------------------------------------------------1')

a = pd.Series([1,2,3,4])
b = a.tolist()
print(b, type(b))

print('\n')
print('------------------------------------------------------------------------------------------2')

x1 = pd.Series([1,2,3,4])
x2 = pd.Series([5,6,7,8])
print(x1*x2,x1+x2,x1-x2,x1/x2)

print('\n')
print('------------------------------------------------------------------------------------------3')

x1 = pd.Series([1,-2,33,4])
x2 = pd.Series([5,6,7,8])
equal = x1==x2
larger = x1>x2
print(equal, larger)

print('\n')
print('------------------------------------------------------------------------------------------4')

x1 = {'a':1,'b':2,'c':3}
x2 = pd.Series(x1)
a = np.array([1,2,3,4])
x3 = pd.Series(a)

print(x2,x3)

print('\n')
print('------------------------------------------------------------------------------------------5')

x1 = pd.Series([1,'mehrsa',2,False,22])
x2 = pd.to_numeric(x1,errors='coerce')
print(x2)

print('\n')
print('------------------------------------------------------------------------------------------6')

df = pd.DataFrame({'a':[1,'mehrsa',2,11,22],'b':[1,'mohsen',2,False,22],'c':[1,'mehrsa',2,False,22]})
print(df)
print(df.iloc[:,0])

print('\n')
print('------------------------------------------------------------------------------------------7')


ds = pd.Series([1,'mehrsa',3,4,5])
print(np.array(ds))

print('\n')
print('------------------------------------------------------------------------------------------8')

ds = pd.Series([[1,'mehrsa',3,4,5],[5,6,7,8]])
ds1 = ds.apply(pd.Series).stack().reset_index(drop = True)
print(ds1)

print('\n')
print('------------------------------------------------------------------------------------------9')
ds = pd.Series([1,122,3,4,5])
ds1 = ds.sort_values()
print(ds1)

print('\n')
print('------------------------------------------------------------------------------------------10')

ds = pd.Series([1,122,3,4,5])
ds1 = pd.Series([74,-13])
ds = ds.append(ds1,ignore_index=True)
print(ds)

print('\n')
print('------------------------------------------------------------------------------------------11')

ds = pd.Series([1,122,3,4,5,77,89])
ds1 = ds[ds<10]
print(ds1)

print('\n')
print('------------------------------------------------------------------------------------------12')

ds = pd.Series([1,122,3,4,5,77,89], index = ['a','b','c','d','e','f','g'])
ds1  = ds.reindex(index = ['b','a','c','d','e','f','g'])
print(ds1)

print('\n')
print('------------------------------------------------------------------------------------------13')

ds = pd.Series([1,122,3,4,5,77,89], index = ['a','b','c','d','e','f','g'])

print(ds.mean(), ds.std())

print('\n')
print('------------------------------------------------------------------------------------------14')

ds = pd.Series([1,122,3,4,5,77,89], index = ['a','b','c','d','e','f','g'])
ds1 = pd.Series([1,-12,3,41,5,77,89])

ds2 = ds[~ds.isin(ds1)]
ds3 = ds[ds.isin(ds1)]
print(ds2,ds3)

print('\n')
print('------------------------------------------------------------------------------------------15')

ds = pd.Series([1,122,3,4,5,77,89], index = ['a','b','c','d','e','f','g'])
ds1 = pd.Series([1,-12,3,41,5,77,89])

ds2 = ds[~ds.isin(ds1)].append(ds1[~ds1.isin(ds)])
print(ds2)

sr11 = pd.Series(np.union1d(ds, ds1))
sr22 = pd.Series(np.intersect1d(ds, ds1))
ds2 = sr11[~sr11.isin(sr22)]
print(ds2)

print('\n')
print('------------------------------------------------------------------------------------------16')

ds = pd.Series([1,122,3,4,5,77,89,1,-9,25,14])
ds1 = ds.quantile([0,0.25,0.5,0.75,1.0])
print(ds2)


print('\n')
print('------------------------------------------------------------------------------------------17')

ds = pd.Series([1,122,3,4,5,77,89], index = ['a','b','c','d','e','f','g'])
ds1 = pd.Series([1,-12,3,41,5,77,89])

ds2 = ds[~ds.isin(ds1)].append(ds1[~ds1.isin(ds)])
print(ds2)

sr11 = pd.Series(np.union1d(ds, ds1))
sr22 = pd.Series(np.intersect1d(ds, ds1))
ds2 = sr11[~sr11.isin(sr22)]
print(ds2)

print('\n')
print('------------------------------------------------------------------------------------------18')

ds = pd.Series([1,122,3,4,5,77,5,2,1,1,5,25,89,1,-9,25,14])
print(ds.value_counts())

print('\n')
print('------------------------------------------------------------------------------------------19')

ds = pd.Series([1,122,3,3,3,4,5,77,5,2,1,1,5,25,89,1,-9,25,14])
print(ds)
ds1 = ds.value_counts()[:2]
ds[~ds.isin(ds1.index)] = 'not max'
print(ds)

print('\n')
print('------------------------------------------------------------------------------------------20')


ds = pd.Series([1,122,3,3,3,4,5,77,5,2,1,1,5,25,89,1,-9,25,14])
ds1 = ds[ds%5==0]

print(ds1,ds1.index)

print('\n')
print('------------------------------------------------------------------------------------------21')


ds = pd.Series([1,122,3,3,3,4,5,77,5,2,1,1,5,25,89,1,-9,25,14])
a = [2,5,4]
ds1 = ds.take(a)
print(ds1)

print('\n')
print('------------------------------------------------------------------------------------------22')


ds = pd.Series([1,122,3,3,3,4,5,77,5,2,1,1,5,25,89,1,-9,25,14])
ds1 = pd.Series([2,5,4,77,400,-9])
a = ds[ds.isin(ds1)].index
print(a.tolist())

print('\n')
print('------------------------------------------------------------------------------------------23')

ds = pd.Series(['mehrsa','mohsen','soudabeh','fatemeh'])
ds1 = ds.map(lambda x: x[0].upper() + x[1:-1] + x[-1].upper())

print(ds1)

print('\n')
print('------------------------------------------------------------------------------------------24')

ds = pd.Series(['mehrsa','mohsen','soudabeh','fatemeh','eh',''])
ds1 = ds.map(lambda x: len(x))

print(ds1)

print('\n')
print('------------------------------------------------------------------------------------------25')

ds = pd.Series([1,122,3,3,3,4,5,77,5,2,1,1,5,25,89,1,-9,25,14])
ds1 = ds.shift(1)-ds
ds2 = ds1.shift(1)-ds1

print(ds2)

ds1 = ds.diff()
ds2 = ds1.diff()
print(ds2)

print('\n')
print('------------------------------------------------------------------------------------------26')

ds = pd.Series(['01 Jan 2015','10-02-2016','20180307','2014/05/06','2016-04-12','2019-04-06T11:20'])
ds1 = pd.to_datetime(ds)

print(ds1)

print('\n')
print('------------------------------------------------------------------------------------------27')

ds = pd.Series(['01 Jan 2015','10-02-2016','20180307','2014/05/06','2016-04-12','2019-04-06T11:20'])
ds = pd.to_datetime(ds)


print(ds1.dt.day_of_year,ds1.dt.day,ds1.dt.dayofweek,ds1.dt.day_name(),ds1.dt.month,ds1.dt.week)

print('\n')
print('------------------------------------------------------------------------------------------28')

ds = pd.Series(['Jan 2015', 'Feb 2016','Mar 2017','Apr 2018','May 2019'])

result = ds.map(lambda d: parse('11 ' + d))
print(result)

ds = pd.to_datetime(ds)
ds = ds.apply(lambda dt: dt.replace(day=11))

print(ds)

print('\n')
print('------------------------------------------------------------------------------------------29')

ds = pd.Series(['mehrsa','mohsen','soudabeh','fatemeh','eh',''])


result = ds.map(lambda c: len([a for a in c if a in 'aeou' ])>=2)
ds1 = ds[result]

print(result)
print(ds1)


print('\n')
print('------------------------------------------------------------------------------------------30')

ds = pd.Series([1,12,3,4,5,6])
ds1 = pd.Series([1,12,3,4,5,8])

result = np.sqrt(sum((ds-ds1)**2))
dist = np.linalg.norm(ds-ds1)

print(result,dist)

print('\n')
print('------------------------------------------------------------------------------------------31')

ds = pd.Series([1,12,3,4,8,3,9])

ds1 = [i for i in ds.index[1:-1] if ds[i]>ds[i-1] and ds[i]>ds[i+1]]

print(ds)
print(ds1)

nums = pd.Series([1, 8, 7, 5, 6, 5, 3, 4, 7, 1])
print("Original series:")
print(nums)
print("\nPositions of the values surrounded by smaller values on both sides:")
temp = np.diff(np.sign(np.diff(nums)))
result = np.where(temp == -2)[0] + 1
print(result)
ds1 = [i for i in nums.index[1:-1] if nums[i]>nums[i-1] and nums[i]>nums[i+1]]

print(ds1)
print('\n')
print('------------------------------------------------------------------------------------------32')

ds = pd.Series([1, 8, 7,'', 5,5,3,7, 6, 5, 3, 4, 7,1, 1])
ds1 = ds.value_counts()
print(ds)
print(ds1.drop(''))
print(ds.replace('',ds1.index[-1]))


ds = pd.Series(list('abc Reyhaneh mehrsa man mehri'))
ds1 = ds.value_counts()
print(ds)
print(ds1.drop(' '))

a = "".join(ds.replace(' ', ds1.index[-1]))
print(a)

print('\n')
print('------------------------------------------------------------------------------------------33')

ds = pd.Series(np.random.rand(10)*20)

print(ds)

autocor = [ds.autocorr(i).round(2) for i in range(len(ds))] 
autocor1 = ds.autocorr(lag = 1)
print(autocor)

print('\n')
print('------------------------------------------------------------------------------------------34')

ds = pd.Series(np.random.rand(10)*20)

print(ds)

autocor = [ds.autocorr(i).round(2) for i in range(len(ds))] 
autocor1 = ds.autocorr(lag = 1)
print(autocor)

print('\n')
print('------------------------------------------------------------------------------------------35')

ds = pd.Series(pd.date_range(start='1/1/2019', end='31/12/2019'))
ds = pd.to_datetime(ds)
ds = ds[ds.dt.dayofweek == 6]
print(ds)


result = pd.Series(pd.date_range('2020-01-01', periods=52, freq='W-SUN'))
print(result)



print('\n')
print('------------------------------------------------------------------------------------------36')

ds = pd.Series(range(3))
ds1 = pd.Series(pd.util.testing.rands_array(2, 3))

df = pd.concat([ds,ds1], axis = 1)
print(df)

print('\n')

print('------------------------------------------------------------------------------------------37')

ds = pd.Series(pd.util.testing.rands_array(1, 300))
ds1 = pd.Series(pd.util.testing.rands_array(1, 300))

ds2 = pd.Series(ds==ds1) 
ds3 = ds2[ds2]
print(ds2,ds3)

print('\n')
print('------------------------------------------------------------------------------------------38')

ds = pd.Series(np.random.rand(20)*10)

print(ds, ds.idxmax(), ds.idxmin(), ds.max(),ds.min())

print('\n')
print('------------------------------------------------------------------------------------------39')

df = pd.DataFrame({'a':[1,2,3],'b':[5,6,3],'c':[7,8,1]})

ds = pd.Series([1,2,3])

df1 = ~df.ne(ds,axis=0) #inverts the ne function results

print(df1)

print('\n')
print('------------------------------------------------------------------------------------------40')
exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

df = pd.DataFrame(exam_data , index=labels)

print('info\n',df.info())
print('shape\n',df.shape)

print('describe\n',df.describe())

print('specific coumns\n',df[['name','attempts']])
print('specific raws\n',df.head(3),df.tail(3))

print('specific raws and columns\n',df.loc[['a','e','f'],['name','attempts']])
print('specific raws and columns\n',df.iloc[[0,4,5],[0,2]])

print('attempts more than 2\n',df[df['attempts']>2])

print('nan scores\n',df[df['score'].isnull()])

print('score between a specific range\n',df[df['score'].between(15, 20)])
print('score between some specific ranges\n',df[(df['attempts']<2) & (df['score']>15)])
print('score between some specific ranges\n',df[(df['attempts']<2) | (df['score']>15)])

df.loc['d','score'] = 11.5
print('chnaged value',df.loc['d','score'])

print('sum of attempts', df['attempts'].sum())
print('average score',df['score'].mean())

df.loc['k'] = ['Mohsen',12,3,'no']
print('new like k is added\n',df)

df = df.drop('k') 
print('new like k is removed\n',df)


df = df.sort_values(by=['name','score'], ascending=[True,False])

print('sorted by name\n',df)

df['qualify'] = df['qualify'].replace(['yes','no'],['True','False'])
print('qualify column replaced\n',df)


df['qualify'] = df['qualify'].map({'True':'yes', 'False':str('no')})
print('qualify column replaced\n',df)

df['name'] = df['name'].replace('James','Soroush')
print('qualify column replaced\n',df)

df = df.drop('attempts',axis = 1)
print('attempts deleted\n',df)

#----------------------------------------------------------------------------------------------------iterate over rows 41:
print('\n')
for index, row in df.iterrows():
    print(index,row['name'], row['score'])

print('\n')

print(df.columns.tolist())


df.columns = ['Column1', 'Column2', 'Column3']
print('columns renamed\n',df)

#--------------------------------------------------------------------------------------------------------------------- 42:
d = {'col1': [1, 4, 3, 4, 5], 'col2': [4, 5, 6, 7, 8], 'col3': [7, 8, 9, 0, 1]}
df = pd.DataFrame(data=d)
print("Original DataFrame")
print(df)
print('Rows for colum1 value == 4')
print(df.loc[df['col1'] == 4])
print(df[df['col1'] == 4])
print('\n')
#---------------------------------------------------------------------------------------------------------------change the column oreder 43:

df = df[['col3', 'col2', 'col1']]

#---------------------------------------------------------------------------------------------------------------write and read from file 44:
df.to_csv('new_file.csv', sep='\t', index=False)
new_df = pd.read_csv('new_file.csv',sep='\t')
print(new_df)
print(df)

print('\n')
#----------------------------------------------------------------------------------------------------count based on a column and rename it 45:
df1 = pd.DataFrame({'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
'city': ['California', 'Los Angeles', 'California', 'California', 'California', 'Los Angeles', 'Los Angeles', 'Georgia', 'Georgia', 'Los Angeles']})

g1 = df1.groupby(["city"]).count().rename(columns ={'name':'number of people'}).reset_index()
print(g1)

g1 = df1.groupby(["city"]).size().reset_index(name='Number of people')
print(g1)

print('\n')
#----------------------------------------------------------------------------------------------------count based on a column and rename it 45:
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
print("Original DataFrame")
print("second row", df.loc[[2]])


#----------------------------------------------------------------------------------------------------count based on a column and rename it 46:
exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

df = pd.DataFrame(exam_data , index=labels)
print(df)
df = df.fillna(0)
print('nan replaced by 0\n',df)

df['index'] = df.index
print('index added into a column\n',df)

df = df.drop('index',axis =1)
print('index column removed\n',df)

df = df.reset_index()
print('reset index\n',df)
df = df.reset_index(drop=True)
print('reset index and droped\n',df)

df = df.set_index('index')
print('new index \n',df)

#---------------------------------------------------------------------------------------------------------------------------------------------------

dummy = pd.get_dummies(so_survey_df, columns=['Country'], drop_first=True, prefix='DM')
mask = countries.isin(country_counts[country_counts<10].index)
new_countries =‌ countries[mask]

# Specify the boundaries of the bins
bins = [-np.inf, 10000, 50000, 100000, 150000, np.inf]

# Bin labels
labels = ['Very low', 'Low', 'Medium', 'High', 'Very high']

# Bin the continuous variable ConvertedSalary using these boundaries
so_survey_df['boundary_binned'] = pd.cut(so_survey_df['ConvertedSalary'], bins, labels)

# Bin the continuous variable only by number of bins
so_survey_df['boundary_binned'] = pd.cut(so_survey_df['ConvertedSalary'], bins = 5)

<<<<<<< HEAD
<<<<<<< HEAD
for label,row in cars.iterrows():
    cars.loc[label,'COUNTRY'] = row['country'].upper()
    
df['column'].cumsum(),   df['column'].sum(),df['column'].mean(),df['column'].mode(),df['column'].std()


#---------------------------------------------------------------------------------------------------------------------------------------------------
def iqr(column):
    return column.quantile(0.75) - column.quantile(0.25)
sales[["temperature_c", "fuel_price_usd_per_l", "unemployment"]].agg([iqr,np.median])
sales_by_type_is_holiday = sales.groupby(["type","is_holiday"])["weekly_sales"].sum()

df.sample(n=5, random_state=1)




#---------------------------------------------------------------------------------------------------------------------------------------------------

# Merge the movies table with the financials table with a left join
movies_financials = movies.merge(financials, left_on='id', right_on='fin_id', how='left'), #how =‌ 'right', 'outer'

# Count the number of rows in the budget column that are missing
number_of_missing_fin = movies_financials['budget'].isnull().sum()

# Print the number of movies missing financials
print(number_of_missing_fin)



#-----------------------------------------------------------------------------------------------A U B - A eshterak B
# Merge iron_1_actors to iron_2_actors on id with outer join using suffixes
iron_1_and_2 = iron_1_actors.merge(iron_2_actors,
                                     on = 'id',
                                     how = 'outer',
                                     suffixes=('_1','_2'))

# Create an index that returns true if name_1 or name_2 are null
m = ((iron_1_and_2['name_1'].isnull()) | 
     (iron_1_and_2['name_2'].isnull()))
     
     
     
     
     
#-----------------------------------------------------------------------------------------------A U B - A eshterak B    
     # Merge the crews table to itself
crews_self_merged = crews.merge(crews, on='id', how='inner',
                                suffixes=('_dir','_crew'))

# Create a boolean index to select the appropriate rows
boolean_filter = ((crews_self_merged['job_dir'] == 'Director') & 
                  (crews_self_merged['job_crew'] != 'Director'))
direct_crews = crews_self_merged[boolean_filter]

# Print the first few rows of direct_crews
print(direct_crews)

# Print the first few rows of iron_1_and_2
print(iron_1_and_2[m].head())

#----------------------------------------------------------------------------------------semi join
# Merge classic_18_19 with pop_18_19
classic_pop = classic_18_19.merge(pop_18_19, on = 'tid')


# Using .isin(), filter classic_18_19 rows where tid is in classic_pop
popular_classic = classic_18_19[classic_18_19['tid'].isin(classic_pop['tid'])]
print(popular_classic)


#------------------------------------------------------------------------------
# Merge employees and top_cust
empl_cust = employees.merge(top_cust, on='srid', 
                                 how='left', indicator=True)

# Select the srid column where _merge is left_on
srid_list = empl_cust.loc[empl_cust['_merge'] == 'left_only','srid']
print(employees[employees['srid'].isin(srid_list)])


#------------------------------------------------------------------------------------------------------------------

gdp_sp500 = pd.merge_ordered(gdp, sp500, left_on='year', right_on='date', 
                             how='left',  fill_method='ffill')

# Subset the gdp and returns columns
gdp_returns = gdp_sp500[['gdp','returns']]

# Print gdp_returns correlation
print (gdp_returns.corr())



#-----------------------------------------------------------------------------------------Select by Query

social_fin.query('value<0 and financial == "net_income"')
# Merge gdp and pop on date and country with fill
gdp_pop = pd.merge_ordered(gdp, pop, on=['country','date'], fill_method='ffill')

# Add a column named gdp_per_capita to gdp_pop that divides the gdp by pop
gdp_pop['gdp_per_capita'] = gdp_pop['gdp'] / gdp_pop['pop']

# Pivot data so gdp_per_capita, where index is date and columns is country
gdp_pivot = gdp_pop.pivot_table('gdp_per_capita', 'date', 'country')

# Select dates equal to or greater than 1991-01-01
recent_gdp_pop = gdp_pivot.query('date>= "1991-01-01"')

# Plot recent_gdp_pop
recent_gdp_pop.plot(rot=90)
plt.show()




#-----------------------------------------------------------------------------------------melt
# unpivot everything besides the year column
ur_tall = ur_wide.melt(id_vars = ['year'], var_name='month', value_name='unempl_rate')


# Create a date column using the month and year columns of ur_tall
ur_tall['date'] = pd.to_datetime(ur_tall['year'] + '-' + ur_tall['month'])

# Sort ur_tall by date in ascending order
ur_sorted = ur_tall.sort_values(by = 'date')

# Plot the unempl_rate by date
ur_sorted.plot(x = 'date',y = 'unempl_rate' )
plt.show()

#-----------------------------------------------------------------------------------------nlargest function: very useful
users_last_10 = set(joined_pr.nlargest(10,'date')['user'])
