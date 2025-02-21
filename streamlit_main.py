import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import plotly.express as px
import random

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
data_path = os.path.join(script_dir, 'data\survery_results_cleaned.csv')
aux_data_path = os.path.join(script_dir, 'data\QUESTION_KEY.csv')

st.set_page_config(layout="wide")

df = pd.read_csv(data_path)
aux_df = pd.read_csv(aux_data_path)


st.markdown('# Can coffee predict your vote during the next elections ?')
st.markdown('## A survey on american coffee consumption habits and political affiliations.')


default_columns = ['What is your age?', 'Gender','What is your favorite coffee drink?', 'How much caffeine do you like in your coffee?', 'Why do you drink coffee?','Political Affiliation']
possible_columns = [col_name for col_name  in df.columns[2:] if len(col_name.split('(')) == 0]
columms = st.multiselect('Select the columns of the data frame you want to display', df.columns[2:], default_columns)

st.dataframe(df[['Submission ID', *columms]])

st.markdown('## Description of the population')

sngl_qst = st.selectbox('Single answers questions', 
                          aux_df[
                              (aux_df['Question Type'] !=  'Text') &
                              (aux_df['Question Type'] != 'Multiple selection') &
                              (aux_df['Question Type'] != 'Number scale') &
                              (aux_df['Question'] != 'Political Affiliation') 
                          ]['Question']
                        )

labels = df[sngl_qst].dropna().unique()
labels.sort()
default_colors = px.colors.qualitative.Bold

default_colors = ["#3c6c37",
"#f2e1c2",
"#7f5000",
"#f29c27",
"#092926",
"#5a773d",
# "#e8d5b7",
"#c0631f",
"#5B5B5B",
# "#1b3d2f"
]

color_map = {label: default_colors[i % len(default_colors)] for i, label in enumerate(sorted(labels))}
# st.write(color_map)
colors = [color_map[label] for label in labels]

left_column, right_column = st.columns(2, gap='large')

# with right_column:
#     vals = df[sngl_qst].value_counts()[labels]
    
#     pie_chat = go.Figure(data=[go.Pie(labels=labels, 
#                                       values=vals, 
#                                       title=sngl_qst, 
#                                       domain={'x': [0, 0.8], 'y': [0.4, 1]}, 
#                                       marker=dict(colors=colors)
#                                     )])
#     pie_chat.update_layout(width=1000, 
#                            height=250, 
#                            margin=dict(l=0, r=0, t=0, b=0),
#                            legend_title="Categories",)
    
#     st.plotly_chart(pie_chat)
    
# with left_column:

#========================================================================================================================================================================
age_cat = ['<18 years old', '18-24 years old', '25-34 years old', '35-44 years old', '45-54 years old', '55-64 years old', '>65 years old']


# ================================================================================================================================================================================

pol_aff = ['Democrat', 'No affiliation', 'Republican', 'Independent']
data_dict = {"Political affiliation" : pol_aff}
for elm in labels:
    count_values = df[df[sngl_qst] == elm]['Political Affiliation'].value_counts()
    for cat in pol_aff:
        count_values.at[cat] = count_values.get(cat, 0)
    count_values = pd.concat([count_values, pd.Series([len(df[df[sngl_qst] == elm])], index=['TOTAL'])])
    # st.write(count_values[[*age_cat, 'Total']])
    data_dict[str(elm)] = count_values[[*pol_aff, 'TOTAL']]
chart_data = pd.DataFrame(data_dict, columns=[*labels])
normalized_data = chart_data.copy()
# st.write(chart_data)
# st.write(normalized_data)
normalized_data.iloc[:, :] = ((normalized_data.iloc[:, :].div(normalized_data.iloc[:, :].sum(axis=1), axis=0)) * 100).map(lambda x: str(x))
fig2 = go.Figure()
for i, col in enumerate(normalized_data.columns): 
    fig2.add_trace(go.Bar(
        x=normalized_data[col], 
        y=normalized_data.index, 
        name=col, 
        orientation='h',
        marker=dict(color=color_map[col])
    ))

fig2.update_layout(barmode='stack',
                xaxis_title=sngl_qst,
                yaxis_title="Age Group",
                 yaxis=dict(
                    tickvals=list(range(len(normalized_data.index))),
                    ticktext=list(normalized_data.index[:-1]) + [f"<b><span style='color:black'>{normalized_data.index[-1]}</b>"]
                ),
                legend_title="Categories",
                width=1000,
                height=300,
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis=dict(tickformat=".0f", range=[0, 100]) 
)
st.plotly_chart(fig2, key=56)

#================================================================================================================================================================================
def format_res(inpt:str)->int:
    splited = inpt.split('-')
    if len(splited) == 1:
        return int(inpt.split('$')[-1])
    return 0.5*sum([int(elm[1:]) for elm in splited])

dumbell_data = {'Age group': pol_aff,
                'Maximum spendings' : [],
                'Actual spendings' : [],
                }
# st.markdown("## Expected expenses vs. Actual ")
for age in pol_aff:
    expenses = df[df['Political Affiliation'] == age]["What is the most you've ever paid for a cup of coffee?"].dropna().map(format_res)
    dumbell_data['Actual spendings'].append(expenses.mean())
    expectations = df[df['Political Affiliation'] == age]["What is the most you'd ever be willing to pay for a cup of coffee?"].dropna().map(format_res)
    dumbell_data['Maximum spendings'].append(expectations.mean())

dumbell_df = pd.DataFrame(dumbell_data)

# st.dataframe(dumbell_data)

# st.markdown

fig = go.Figure()

for i, row in dumbell_df.iterrows():
    fig.add_trace(go.Scatter(
        x=[row["Age group"], row["Age group"]],
        y=[row["Actual spendings"], row["Maximum spendings"]],  
        mode="lines",
        line=dict(color="grey"),
        showlegend=False
    ))
# st.write([f"{val:.2f}" for val in dumbell_data["Actual spendings"]])
fig.add_trace(go.Scatter(
    x=dumbell_data["Age group"],
    y=dumbell_data["Actual spendings"],
    mode="markers+text",
    name="Actual spendings",
    marker=dict(color=default_colors[0]),
    text=[f"{val:.2f}" for val in dumbell_data["Actual spendings"]], 
    textposition="middle right"
))

# Add Expectation points
fig.add_trace(go.Scatter(
    x=dumbell_data["Age group"],
    y=dumbell_data["Maximum spendings"],
    mode="markers+text",
    name="Maximum spendings",
    marker=dict(color=default_colors[1]),
    text=[f"{val:.2f}" for val in dumbell_data["Maximum spendings"]],
    textposition="bottom right"
))

fig.update_layout(
    title="Real average spendings on a cup of coffee vs maximum spendings, by age group",
    xaxis_title="Age Group",
    yaxis_title="Price",
    yaxis=dict(range=[0, 1.15*dumbell_df[["Actual spendings", "Maximum spendings"]].max().max()])  
)

st.plotly_chart(fig)

left_col, right_col = st.columns([1,2])

with left_col:

    mlti_qst = st.selectbox('Multiple asnwers questions',
                            aux_df[
                                (aux_df['Question Type'] == 'Multiple selection') &
                                (aux_df['Question'] != 'What kind of flavorings do you add?')
                            ]['Question']
                        )

# mlti_answ = list(map(lambda x: x.strip(), aux_df[aux_df['Question'] == mlti_qst]['Answer Choices'].to_list()[0].split(',')))



counts = df[mlti_qst].value_counts()
total_counts = counts.sum() 
top_labels = counts.nlargest(5).to_frame(name="Count")  
top_labels["Percentage"] = (top_labels["Count"] / total_counts * 100).map(lambda x: f"{x:.2f}%") 
max_val = top_labels["Count"].max()

flp_labels = counts[counts > 20].nsmallest(5).to_frame(name="Count")
flp_labels["Percentage"] = (flp_labels["Count"] / total_counts * 100).map(lambda x: f"{x:.2f}%") 

show_border = True
show_border = False
left_col, right_col = st.columns([1,1], gap='small', border=show_border)

def color_scale_red(row):
    """Color the entire row based on the 'Score' column, scaling from green (low) to red (high)."""
    score = float(row['Count'])/float(max_val) 
    color = f'background-color: rgb({int(322+64 *(score))}, 0, 0)' 
    return ['', color] 

def color_scale_green(row):
    """Color the entire row based on the 'Score' column, scaling from green (low) to red (high)."""
    score = float(row['Count'])/float(max_val) 
    color = f'background-color: rgb(0, {int( 128+128*(score))}, 0)' 
    return ['', color] 

top_lb_sty = top_labels.style.apply(color_scale_green, axis=1)
flp_lb_sty = flp_labels.style.apply(color_scale_red, axis=1)
with left_col : 
    st.markdown("<p style='text-align: center; font-size: 40px;'>5 TOPS </p>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 6, 1], border=show_border, gap='small')
    c2.dataframe(top_lb_sty, width=10000)
with right_col:
    st.markdown("<p style='text-align: center; font-size: 40px;'>5 FLOPS </p>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 6, 1], border=show_border, gap='small')
    c2.dataframe(flp_lb_sty, width=10000)

st.markdown('## Coffee evaluation')

rank = ['Bitterness', 'Acidity', 'Personal Preference']
c1, c2 = st.columns([1, 1], gap='small')
cfs = [['Coffee A', 'Coffee C'], ['Coffee B', 'Coffee D']]



for i, clm in enumerate([c1, c2]) :
    with clm:
        coffees = cfs[i]
        for cf in coffees:
            clm.markdown(f"<p style='text-align: center; font-size: 30px;'> {cf}</p>", unsafe_allow_html=True)
            cc1, cc2 = st.columns([1, 5], gap='small', vertical_alignment='top')
            fig = go.Figure()
            ranks = [np.average(df[c].dropna()) for c in [f'{cf} - {crit}' for crit in rank]]
            res_df = pd.DataFrame({'Criterion' : rank, 'Ranks': ranks})
            fig.add_trace(go.Scatterpolar(
                            r=ranks,
                            theta=rank,
                            fill='toself'
                        )
            )
            fig.update_layout(
                polar=dict(
                    bgcolor="#e8eaf5",
                    radialaxis=dict(
                        visible=True,
                        range=[0, 5]
                    )),
                showlegend=False,
                width=400,
                height=400,
                margin=dict(l=100, r=50, t=0, b=20),
                )
            cc2.plotly_chart(fig)
