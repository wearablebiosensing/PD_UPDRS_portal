
df_plot_sessions = pd.read_csv("/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_IOTEX/iotex-glove/PD/Participant1/rise_times.csv")    
fig = px.histogram(df_plot_sessions, x="percentile_list", y="rise_time_arr_seconds",
             color='med_status', barmode='group',
             histfunc='avg',
             height=400,text_auto=True)
fig.update_layout(
   xaxis = dict(
      tickmode = 'linear',
      #tick0 = 0.5,
      #dtick = 0.75
   )
)        
fig.show()