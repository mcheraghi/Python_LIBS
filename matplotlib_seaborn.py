
#--------------------------------------------------------------------------------Very basic plot
fig,ax = plt.subplots()
ax.plot(seattle_weather["MONTH"], seattle_weather["MLY-PRCP-NORMAL"],marker = 'o',color = 'b',linestyle = '--')
ax.plot(austin_weather["MONTH"], austin_weather["MLY-PRCP-NORMAL"])

# Customize the x-axis label
ax.set_xlabel("Time (months)")

# Customize the y-axis label
ax.set_ylabel("Precipitation (inches)")

# Add the title
ax.set_title("Weather patterns in Austin and Seattle")

# Display the figure
plt.show()


#------------------------------------------------------------------------------------
fig, ax = plt.subplots(m,n)   #m rows and n columns, ax is an array of shape (m,n)


#------------------------------------------------------------------------------------
import pandas as pd

# Read the data from file using read_csv
climate_change0 = pd.read_csv('climate_change.csv', parse_dates = True, index_col = 'date')
fig, ax = plt.subplots()

climate_change = climate_change0["1970-01-01":"1979-12-31"]
# Add the time-series for "relative_temp" to the plot
ax.plot(climate_change.index,climate_change["relative_temp"])

# Set the x-axis label
ax.set_xlabel('Time')

# Set the y-axis label
ax.set_ylabel('Relative temperature (Celsius)')

# Show the figure
plt.show()


#------------------------------------------------------------------------------------twinx()

import matplotlib.pyplot as plt

# Initalize a Figure and Axes
fig,ax = plt.subplots()

# Plot the CO2 variable in blue
ax.plot(climate_change.index, climate_change['co2'], color='b')

# Create a twin Axes that shares the x-axis
ax2 = ax.twinx()

# Plot the relative temperature in red
ax2.plot(climate_change.index, climate_change["relative_temp"], color='r')

plt.show()

#------------------------------------------------------------------------------------basic plot function()
# Define a function called plot_timeseries
def plot_timeseries(axes, x, y, color, xlabel, ylabel):

  # Plot the inputs x,y in the provided color
  axes.plot(x,y, color=color)

  # Set the x-axis label
  axes.set_xlabel(xlabel)

  # Set the y-axis label
  axes.set_ylabel(ylabel, color=color)

  # Set the colors tick params for y-axis
  axes.tick_params('y', colors=color)
  
  
#------------------------------------------------------------------------------------ax.Annotate(): putting an arrow and text
ax2.annotate(">1 degree", xy = (pd.Timestamp('2015-10-06'),1),xytext = (pd.Timestamp('2008-10-06'),-0.2) , arrowprops=dict(arrowstyle="->",color = 'gray'))

#----------------------------------------------
fig, ax = plt.subplots()
# Add bars for "Gold" with the label "Gold"
ax.bar(medals.index, medals['Gold'], label="Gold")

# Stack bars for "Silver" on top with label "Silver"
ax.bar(medals.index, medals['Silver'], bottom=medals['Gold'], label = 'Silver')

# Stack bars for "Bronze" on top of that with label "Bronze"
ax.bar(medals.index,medals['Bronze'],bottom = medals['Gold'] + medals['Silver'],label = 'Bronze')

# Display the legend
ax.legend()

plt.show()



#-----------------------------------------------------------Histogram
fig, ax = plt.subplots()

# Plot a histogram of "Weight" for mens_rowing
ax.hist( mens_rowing['Weight'],label = "Rowing", bins = 5, histtype = 'step')

# Compare to histogram of "Weight" for mens_gymnastics
ax.hist(mens_gymnastics['Weight'],label = "Gymnastics", bins = 5,histtype = 'step')

ax.set_xlabel("Weight (kg)")
ax.set_ylabel("# of observations")

# Add the legend and show the Figure
ax.legend()

#--------------------------------------------------------Errorbars and boxplot
fig, ax = plt.subplots()

# Add a bar for the rowing "Height" column mean/std
ax.bar("Rowing",mens_rowing['Height'].mean() , yerr= mens_rowing['Height'].std())

# Add a bar for the gymnastics "Height" column mean/std
ax.bar("Gymnastics",mens_gymnastics['Height'].mean() , yerr= mens_gymnastics['Height'].std())

# Label the y-axis
ax.set_ylabel("Height (cm)")

plt.show()


#------------------------------------------------------
fig, ax = plt.subplots()

# Add Seattle temperature data in each month with error bars
ax.errorbar(seattle_weather["MONTH"], seattle_weather["MLY-TAVG-NORMAL"], seattle_weather["MLY-TAVG-STDDEV" ])

# Add Austin temperature data in each month with error bars
ax.errorbar(austin_weather["MONTH"], austin_weather["MLY-TAVG-NORMAL"], austin_weather["MLY-TAVG-STDDEV" ]) 

# Set the y-axis label
ax.set_ylabel("Temperature (Fahrenheit)")

plt.show()

#-------------------------------------------------------------

fig, ax = plt.subplots()

# Add a boxplot for the "Height" column in the DataFrames
ax.boxplot([mens_rowing["Height"],mens_gymnastics["Height"]])

# Add x-axis tick labels:
ax.set_xticklabels(['Rowing','Gymnastics'])

# Add a y-axis label
ax.set_ylabel("Height (cm)")

plt.show()



#----------------------------------------------------------------------------------------------------

fig, ax = plt.subplots()

# Add data: "co2", "relative_temp" as x-y, index as color
a = ax.scatter(climate_change['co2'],climate_change["relative_temp"],c=climate_change.index)

# Set the x-axis label to "CO2 (ppm)"
ax.set_xlabel("CO2 (ppm)")
plt.colorbar(a)

# Set the y-axis label to "Relative temperature (C)"
ax.set_ylabel("Relative temperature (C)")

plt.show()


#----------------------------------------------------------------------------------------------------Plot Styles$
# Use the "Solarize_Light2" style and create new Figure/Axes
plt.style.use('Solarize_Light2') #can be 'ggplot' and many others
fig,ax= plt.subplots()
ax.plot(austin_weather["MONTH"], austin_weather["MLY-TAVG-NORMAL"])
plt.show()
fig.set_size_inches([5,3])
fig.savefig("my_figure_300dpi.png",dpi=300)
# Set figure dimensions and save as a PNG


#---------------------------------------------------------------
sports = summer_2016_medals['Sport'].unique()
fig, ax = plt.subplots()

# Loop over the different sports branches
for sport in sports:
  # Extract the rows only for this sport
  sport_df = summer_2016_medals[summer_2016_medals['Sport'] == sport]
  # Add a bar for the "Weight" mean with std y error bar
  ax.bar(sport,sport_df['Weight'].mean(),yerr = sport_df['Weight'].std())

ax.set_ylabel("Weight")
ax.set_xticklabels(sports, rotation=90)

# Save the figure to file
fig.savefig("sports_weights.png")




#----------------------------------------------------------------------------------------------sns
# Import Matplotlib and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Change the legend order in the scatter plot
sns.scatterplot(x="absences", y="G3", 
                data=student_data, 
                hue="location",
                hue_order= ['Rural',"Urban"])

# Show plot
plt.show()

#----------------------------------------------------------------------------
# Create a dictionary mapping subgroup values to colors
palette_colors = {"Rural": "green", "Urban": "blue"}

# Create a count plot of school with location subgroups
sns.countplot(x="school",data=student_data, hue="location",palette=palette_colors)



# Display plot
plt.show()




#----------------------------------------------------------------------------
# Adjust further to add subplots based on family support
sns.relplot(x="G1", y="G3", 
            data=student_data,
            kind="scatter", 
            col="schoolsup",
            col_order=["yes", "no"],
            row = "famsup",
            row_order=["yes", "no"])

# Show plot
plt.show()



#----------------------------------------------------------------------------
# Import Matplotlib and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Create scatter plot of horsepower vs. mpg

sns.relplot(x="horsepower",y="mpg",kind = 'scatter',data = mpg ,style="origin",size="cylinders",hue="cylinders")


# Show plot
plt.show()


#----------------------------------------------------------------------------relplot(kind='line')
# Import Matplotlib and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Create line plot
sns.relplot(x="model_year", y="mpg",
            data=mpg, kind="line",ci='sd')

sns.relplot(x="model_year", y="horsepower", 
            data=mpg, kind="line", 
            ci=None, style="origin", 
            hue="origin",dashes=False,markers= True)


#----------------------------------------------------------------------------
#sns.scatterplot(x= ..., y = ...) sns.countplot(x= ...) sns.countplot(y = ...)
# Create a distplot
sns.distplot(df['Award_Amount'],
             kde=False,
             bins=20)


#-------------distplot
# Create a distplot of the Award Amount
sns.distplot(df['Award_Amount'],
             hist=False,
             rug=True,
             kde_kws={'shade':True})
             
#----------------------
# Create an lmplot of premiums vs. insurance_losses
sns.lmplot(x="insurance_losses",y="premiums",data=df)


#----------------------
# Create a regression plot using hue
sns.lmplot(data=df,
           x="insurance_losses",
           y="premiums",
           hue="Region")  # hue, row,col
           
           
#----------------------
sns.set()     
sns.set_style('dark')    # ['white','dark',whitegrid','darkgrid','ticks']

df['fmr_2'].plot.hist()





#------------------------
# Set the style to white
sns.set_style('white')

# Create a regression plot
sns.lmplot(data=df,
           x='pop2010',
           y='fmr_2')

# Remove the spines, the right and top borders of the plot
sns.despine()    




#-------------------------
# Set style, enable color code, and create a magenta distplot
sns.set(color_codes=True)
sns.distplot(df['fmr_3'], color='m')   

# this method helps to see the color palette
sns.palplot(sns.color_palette('Purples',8))
plt.show()
   
   
   
   
   
   
#-------------------------------------using seaborn and matplotlib together
# Create a figure and axes
fig, ax = plt.subplots()

# Plot the distribution of data
sns.distplot(df['fmr_3'], ax=ax)

# Create a more descriptive x axis label
ax.set(xlabel="3 Bedroom Fair Market Rent",xlabel='mmm',xlim=(100,1500),title='mmm1')

# Show the plot
plt.show()






#------------------------------------
# Create a figure and axes. Then plot the data
fig, ax = plt.subplots()
sns.distplot(df['fmr_1'], ax=ax)

# Customize the labels and limits
ax.set(xlabel="1 Bedroom Fair Market Rent", xlim=(100,1500), title="US Rent")

# Add vertical lines for the median and mean
ax.axvline(x=df['fmr_1'].median(), color='m', label='Median', linestyle='--', linewidth=2)
ax.axvline(x=df['fmr_1'].mean(), color='b', label='Mean', linestyle='-', linewidth=2)

# Show the legend and plot the data
ax.legend()
plt.show()




#------------------------------------subplot
# Create a plot with 1 row and 2 columns that share the y axis label
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharey=True)

# Plot the distribution of 1 bedroom apartments on ax0
sns.distplot(df['fmr_1'], ax=ax0)
ax0.set(xlabel="1 Bedroom Fair Market Rent", xlim=(100,1500))

# Plot the distribution of 2 bedroom apartments on ax1
sns.distplot(df['fmr_2'], ax=ax1)
ax1.set(xlabel="2 Bedroom Fair Market Rent", xlim=(100,1500))

# Display the plot
plt.show()



#--------------------------------------------------stripplot()
# Create the stripplot
sns.stripplot(data=df,
         x='Award_Amount',
         y='Model Selected',
         jitter=True)

plt.show()


#--------------------------------------------------swarmplot()
# Create and display a swarmplot with hue set to the Region
sns.swarmplot(data=df,
         x='Award_Amount',
         y='Model Selected',
        hue='Region')

plt.show()


#------------------------------------------------boxplot()
# Create a boxplot
sns.boxplot(data=df,
         x='Award_Amount',
         y='Model Selected')

plt.show()

#------------------------------------------------boxplot()

