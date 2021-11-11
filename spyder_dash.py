# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt

app = dash.Dash(__name__)

# let's change the colors 
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

df = pd.read_csv("weather_data.csv")
print(df.columns)

##let's group this data by year to make it more digestible 
df=df.groupby('Year').agg('mean')

# let's also drop any NAs which might be in the data
df.dropna(inplace=True)

# and set our year as a column and not just as an index (groupby will automatically set your grouping factor as the index)
df.reset_index(inplace=True)


fig = px.bar(df, x="Year", y="SALEM_TEMP", barmode="group")

fig2 = px.scatter(df, x='SF_TEMP', y='SF_WIND', text='Year')

# add these updates to your figure
fig.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)


# what if we want multiple figures in our dashboard? We can do that too! (wonderful) 
app.layout = html.Div(children=[
    # now you are creating multiple "children" plots from the top, so add for each plot a new html.Div: 
    html.Div([
        html.H1(
            children='Temperatures in Salem'
        ),

        html.Div(
            children='''
            As you can see, average temperatures in Salem fluctuate usually between 10 and 13 degrees Celsius.
        '''
        ),

        dcc.Graph(
            id='example-graph',
            figure=fig
        ),
    ]),

# just add them in sequence adding a new "html.Div" line, and they will appear below the next plot
    html.Div(children=[
        html.H1(children='Temperature versus Wind in SF'),

        html.Div(children='''
            Can you see any relationship?
        '''
        ),

        dcc.Graph(
            id='example2',
            figure=fig2
        ),
    ]),
])
# make sure everything is indented properly! 
                      
if __name__ == '__main__':
    app.run_server(dev_tools_hot_reload=False)