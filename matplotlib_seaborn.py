
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


#------------------------------------------------------------------------------------
