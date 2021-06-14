import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None

df1 = pd.read_csv('experimentos/fichero_SinMA.csv', sep='\t')
#df2 = pd.read_csv('experimentos/fichero_ConMA.csv', sep='\t')
df1 = df1.drop(columns=['Unnamed: 0'])
#df2 = df2.drop(columns=['Unnamed: 0'])

t = 0.0
for i in range(len(df1['time'])):
    df1['time'][i] = t
    t += 5.0/60.0

#t = 58.0
#for i in range(len(df2['time'])):
#    df2['time'][i] = t
#    t += 5.0/60.0

fig, axs = plt.subplots(2, 2, figsize=(20, 10))

axs[0, 0].plot(df1['time'], df1['Ca'], 'b', label='Ca')
axs[0, 0].plot(df1['time'], df1['state[1]'], 'k', label='Ca_MHE')
#axs[0, 0].plot(df2['time'], df2['Ca'], 'b')
#axs[0, 0].plot(df2['time'], df2['state[1]'], 'k')
axs[0, 0].set_ylim(0.0, 0.10)
axs[0, 0].legend()
axs[0, 0].text(0.13, 0.85, s='No MA', transform=plt.gcf().transFigure)
#axs[0, 0].text(0.13, 0.85, s='MA\nNLMS', transform=plt.gcf().transFigure)
axs[0, 0].set_title('Concentration of A')
axs[0, 0].set_ylabel('Concentration [mol/L]')
axs[0, 0].set_xlabel('Time [min]')

axs[0, 1].plot(df1['time'], df1['Cb'], 'b', label='Cb')
axs[0, 1].plot(df1['time'], df1['state[2]'], 'k', label='Cb_MHE')
#axs[0, 1].plot(df2['time'], df2['Cb'], 'b')
#axs[0, 1].plot(df2['time'], df2['state[2]'], 'k')
axs[0, 1].set_ylim(2.0, 3.5)
axs[0, 1].legend()
axs[0, 1].text(0.55, 0.85, s='No MA', transform=plt.gcf().transFigure)
#axs[0, 1].text(0.55, 0.85, s='MA\nNLMS', transform=plt.gcf().transFigure)
axs[0, 1].set_title('Concentration of B')
axs[0, 1].set_ylabel('Concentration [mol/L]')
axs[0, 1].set_xlabel('Time [min]')

axs[1, 0].plot(df1['time'], df1['T'], 'b', label='T')
axs[1, 0].plot(df1['time'], df1['state[3]'], 'k', label='T_MHE')
#axs[1, 0].plot(df2['time'], df2['T'], 'b')
#axs[1, 0].plot(df2['time'], df2['state[3]'], 'k')
axs[1, 0].set_ylim(20, 50)
axs[1, 0].legend()
axs[1, 0].text(0.15, 0.42, s='No MA', transform=plt.gcf().transFigure)
#axs[1, 0].text(0.15, 0.42, s='MA\nNLMS', transform=plt.gcf().transFigure)
axs[1, 0].set_title('Reactor Temperature')
axs[1, 0].set_ylabel('Temperature [ºC]')
axs[1, 0].set_xlabel('Time [min]')

axs[1, 1].plot(df1['time'], df1['Tc'], 'b', label='Tc')
axs[1, 1].plot(df1['time'], df1['state[4]'], 'k', label='Tc_MHE')
#axs[1, 1].plot(df2['time'], df2['Tc'], 'b')
#axs[1, 1].plot(df2['time'], df2['state[4]'], 'k')
axs[1, 1].set_ylim(20, 30)
axs[1, 1].legend()
axs[1, 1].text(0.57, 0.42, s='No MA', transform=plt.gcf().transFigure)
#axs[1, 1].text(0.57, 0.42, s='MA\nNLMS', transform=plt.gcf().transFigure)
axs[1, 1].set_title('Coolant Temperature')
axs[1, 1].set_ylabel('Temperature [ºC]')
axs[1, 1].set_xlabel('Time [min]')

for i in range(2):
    for j in range(2):
        axs[i, j].set_xlim(0, 58)
        #axs[i, j].axvline(58.0, c='0.8', ls='--')
plt.savefig('experimentos/Figura_1_sinMA.pdf')

fig, axs = plt.subplots(2, 2, figsize=(20, 10))

axs[0, 0].plot(df1['time'], df1['J_costo'], 'b', label='J_cost')
#axs[0, 0].plot(df2['time'], df2['J_costo'], 'b')
axs[0, 0].set_ylim(0.0, 50.0)
axs[0, 0].text(0.13, 0.85, s='No MA', transform=plt.gcf().transFigure)
#axs[0, 0].text(0.13, 0.85, s='MA\nNLMS', transform=plt.gcf().transFigure)
axs[0, 0].set_title('Objective Function')
axs[0, 0].set_ylabel('Profit [€/min]')
axs[0, 0].set_xlabel('Time [min]')

axs[0, 1].plot(df1['time'], df1['Lambda[1]'], 'b', label='Lambda[1]')
#axs[0, 1].plot(df2['time'], df2['Lambda[1]'], 'b')
ax2 = axs[0, 1].twinx()
ax2.plot(df1['time'], df1['Lambda[2]'], 'k', label='Lambda[2]')
#ax2.plot(df2['time'], df2['Lambda[2]'], 'k')
axs[0, 1].text(0.55, 0.85, s='No MA', transform=plt.gcf().transFigure)
#axs[0, 1].text(0.55, 0.85, s='MA\nNLMS', transform=plt.gcf().transFigure)
axs[0, 1].set_title('Modifiers')
axs[0, 1].set_ylabel('Lambda[1]')
axs[0, 1].set_ylim(-50, 50)
ax2.set_ylabel('Lambda[2]')
ax2.set_ylim(-5, 5)
lines, labels = axs[0, 1].get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)
axs[0, 1].set_xlabel('Time [min]')

axs[1, 0].plot(df1['time'], df1['q'], 'b', label='q')
axs[1, 0].plot(df1['time'], df1['uq[1]'], 'k', label='uq')
#axs[1, 0].plot(df2['time'], df2['q'], 'b')
#axs[1, 0].plot(df2['time'], df2['uq[1]'], 'k')
axs[1, 0].set_ylim(0.3, 1.5)
axs[1, 0].legend()
axs[1, 0].text(0.13, 0.4, s='No MA', transform=plt.gcf().transFigure)
#axs[1, 0].text(0.13, 0.42, s='MA\nNLMS', transform=plt.gcf().transFigure)
axs[1, 0].set_title('Reactants Flowrate')
axs[1, 0].set_ylabel('Flow [L/min]')
axs[1, 0].set_xlabel('Time [min]')

axs[1, 1].plot(df1['time'], df1['Fr'], 'b', label='Fr')
axs[1, 1].plot(df1['time'], df1['uFr[1]'], 'k', label='uFr')
#axs[1, 1].plot(df2['time'], df2['Fr'], 'b')
#axs[1, 1].plot(df2['time'], df2['uFr[1]'], 'k')
axs[1, 1].set_ylim(0.83, 15.0)
axs[1, 1].legend()
axs[1, 1].text(0.55, 0.42, s='No MA', transform=plt.gcf().transFigure)
#axs[1, 1].text(0.55, 0.42, s='MA\nNLMS', transform=plt.gcf().transFigure)
axs[1, 1].set_title('Coolant Flowrate')
axs[1, 1].set_ylabel('Flow [L/min]')
axs[1, 1].set_xlabel('Time [min]')

for i in range(2):
    for j in range(2):
        axs[i, j].set_xlim(0, 58)
        #axs[i, j].axvline(58.0, c='0.8', ls='--')

plt.savefig('experimentos/Figura_2_sinMA.svg')
plt.show()
#fig, axs = plt.subplots(2, 2, sharex=True, figsize=(20, 10))