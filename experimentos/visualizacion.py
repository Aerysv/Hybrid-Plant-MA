import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None

df = pd.read_csv('experimentos/fichero _1erExperimento_NoMA_MA.csv', sep='\t')
df = df.drop(columns=['Unnamed: 0'])

t = 0.0
for i in range(len(df['time'])):
    df['time'][i] = t
    t += 5.0/60.0

fig, axs = plt.subplots(2, 2, figsize=(20, 10))

axs[0, 0].plot(df['time'], df['Ca'], 'b', label='Ca')
axs[0, 0].plot(df['time'], df['state[1]'], 'k', label='Ca_MHE')
axs[0, 0].legend()
#axs[0, 0].text(0.13, 0.85, s='MA\nexact gradients', transform=plt.gcf().transFigure)
#axs[0, 0].text(0.2, 0.85, s='No MA', transform=plt.gcf().transFigure)
#axs[0, 0].text(0.31, 0.85, s='MA\nNLMS', transform=plt.gcf().transFigure)
axs[0, 0].text(0.13, 0.85, s='No MA', transform=plt.gcf().transFigure)
axs[0, 0].text(0.27, 0.85, s='MA\nNLMS', transform=plt.gcf().transFigure)
axs[0, 0].set_title('Concentration of A')
axs[0, 0].set_ylabel('Concentration [mol/L]')
axs[0, 0].set_xlabel('Time [min]')

axs[0, 1].plot(df['time'], df['Cb'], 'b', label='Cb')
axs[0, 1].plot(df['time'], df['state[2]'], 'k', label='Cb_MHE')
axs[0, 1].legend()
#axs[0, 1].text(0.55, 0.85, s='MA\nexact gradients', transform=plt.gcf().transFigure)
#axs[0, 1].text(0.62, 0.85, s='No MA', transform=plt.gcf().transFigure)
#axs[0, 1].text(0.73, 0.85, s='MA\nNLMS', transform=plt.gcf().transFigure)
axs[0, 1].text(0.55, 0.85, s='No MA', transform=plt.gcf().transFigure)
axs[0, 1].text(0.69, 0.85, s='MA\nNLMS', transform=plt.gcf().transFigure)
axs[0, 1].set_title('Concentration of B')
axs[0, 1].set_ylabel('Concentration [mol/L]')
axs[0, 1].set_xlabel('Time [min]')

axs[1, 0].plot(df['time'], df['T'], 'b', label='T')
axs[1, 0].plot(df['time'], df['state[3]'], 'k', label='T_MHE')
axs[1, 0].legend()
axs[1, 0].text(0.15, 0.42, s='No MA', transform=plt.gcf().transFigure)
axs[1, 0].text(0.29, 0.42, s='MA\nNLMS', transform=plt.gcf().transFigure)
axs[1, 0].set_title('Reactor Temperature')
axs[1, 0].set_ylabel('Temperature [ºC]')
axs[1, 0].set_xlabel('Time [min]')

axs[1, 1].plot(df['time'], df['Tc'], 'b', label='Tc')
axs[1, 1].plot(df['time'], df['state[4]'], 'k', label='Tc_MHE')
axs[1, 1].legend()
axs[1, 1].text(0.57, 0.42, s='No MA', transform=plt.gcf().transFigure)
axs[1, 1].text(0.71, 0.42, s='MA\nNLMS', transform=plt.gcf().transFigure)
axs[1, 1].set_title('Coolant Temperature')
axs[1, 1].set_ylabel('Temperature [ºC]')
axs[1, 1].set_xlabel('Time [min]')

for i in range(2):
    for j in range(2):
        axs[i, j].set_xlim(0, 140)
        #axs[i, j].axvline(36.4, c='0.8', ls='--')
        #axs[i, j].axvline(91.11, c='0.8', ls='--')
        axs[i, j].axvline(54.71, c='0.8', ls='--')
plt.savefig('Figura_1.pdf')
#plt.show()

fig, axs = plt.subplots(2, 2, figsize=(20, 10))

axs[0, 0].plot(df['time'], df['J_costo'], 'b', label='J_cost')
#axs[0, 0].text(0.13, 0.85, s='MA\nexact gradients', transform=plt.gcf().transFigure)
#axs[0, 0].text(0.2, 0.85, s='No MA', transform=plt.gcf().transFigure)
#axs[0, 0].text(0.31, 0.85, s='MA\nNLMS', transform=plt.gcf().transFigure)
axs[0, 0].text(0.13, 0.85, s='No MA', transform=plt.gcf().transFigure)
axs[0, 0].text(0.27, 0.85, s='MA\nNLMS', transform=plt.gcf().transFigure)
axs[0, 0].set_title('Objective Function')
axs[0, 0].set_ylabel('Profit [€/min]')
axs[0, 0].set_xlabel('Time [min]')

axs[0, 1].plot(df['time'], df['Lambda[1]'], 'b', label='Lambda[1]')
ax2 = axs[0, 1].twinx()
ax2.plot(df['time'], df['Lambda[2]'], 'k', label='Lambda[2]')
#axs[0, 1].text(0.55, 0.85, s='MA\nexact gradients', transform=plt.gcf().transFigure)
#axs[0, 1].text(0.62, 0.85, s='No MA', transform=plt.gcf().transFigure)
#axs[0, 1].text(0.73, 0.85, s='MA\nNLMS', transform=plt.gcf().transFigure)
axs[0, 1].text(0.55, 0.85, s='No MA', transform=plt.gcf().transFigure)
axs[0, 1].text(0.69, 0.85, s='MA\nNLMS', transform=plt.gcf().transFigure)
axs[0, 1].set_title('Modifiers')
axs[0, 1].set_ylabel('Lambda[1]')
axs[0, 1].set_ylim(-50, 50)
ax2.set_ylabel('Lambda[2]')
ax2.set_ylim(-6,6)
lines, labels = axs[0, 1].get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)
axs[0, 1].set_xlabel('Time [min]')

axs[1, 0].plot(df['time'], df['q'], 'b', label='q')
axs[1, 0].plot(df['time'], df['uq[1]'], 'k', label='uq')
axs[1, 0].legend()
axs[1, 0].text(0.13, 0.4, s='No MA', transform=plt.gcf().transFigure)
axs[1, 0].text(0.27, 0.42, s='MA\nNLMS', transform=plt.gcf().transFigure)
axs[1, 0].set_title('Reactants Flowrate')
axs[1, 0].set_ylabel('Flow [L/min]')
axs[1, 0].set_xlabel('Time [min]')

axs[1, 1].plot(df['time'], df['Fr'], 'b', label='Fr')
axs[1, 1].plot(df['time'], df['uFr[1]'], 'k', label='uFr')
axs[1, 1].legend()
axs[1, 1].text(0.55, 0.42, s='No MA', transform=plt.gcf().transFigure)
axs[1, 1].text(0.69, 0.42, s='MA\nNLMS', transform=plt.gcf().transFigure)
axs[1, 1].set_title('Coolant Flowrate')
axs[1, 1].set_ylabel('Flow [L/min]')
axs[1, 1].set_xlabel('Time [min]')

for i in range(2):
    for j in range(2):
        axs[i, j].set_xlim(0, 140)
        #axs[i, j].axvline(36.4, c='0.8', ls='--')
        #axs[i, j].axvline(91.11, c='0.8', ls='--')
        axs[i, j].axvline(54.71, c='0.8', ls='--')

plt.savefig('Figura_2.pdf')
plt.show()
#fig, axs = plt.subplots(2, 2, sharex=True, figsize=(20, 10))