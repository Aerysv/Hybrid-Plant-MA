from influxdb import InfluxDBClient
import pandas as pd

db = InfluxDBClient(host='localhost', port=8086)

tags = '"q", "Fr", "Ca", "Cb", "T", "T0", "Tc", "Tc0", "uq[1]", "uFr[1]", "Lambda[1]", "Lambda[2]", "state[1]", "state[2]", "state[3]", "state[4]", "v_new[1]", "v_new[2]", "v_new[3]", "v_new[4]"'
query = 'SELECT ' + tags + ' FROM "telegraf"."autogen"."eMPC MA" WHERE time >= \'2021-05-31T08:26:00Z\' AND time <= \'2021-05-31T09:28:00Z\''
results = db.query(query)
points = results.get_points()

df = pd.DataFrame(points)
df.set_index('time')
df['J_costo'] = df["q"]*(18.0*df["Cb"] - 0.2*5.0) - df["Fr"]*3.0
print(df)
df.to_csv('fichero_SinMA.csv', index='False', sep='\t')