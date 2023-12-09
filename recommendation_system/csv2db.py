import config
import pandas as pd
from pymysql import connections, cursors
import numpy as np
db_conn = connections.Connection(
    host=config.customhost,
    port=3306,
    user=config.customuser,
    password=config.custompass,
    connect_timeout=600,
    db=config.customdb
)
csv = pd.read_csv(config.RECIPES_DETAILS)
cursor = db_conn.cursor()
null = None

csv["serves"] = csv["serves"].replace(np.nan, null)
csv["cooking_time"] = csv["cooking_time"].replace(np.nan, null)
csv["ingredients"] = csv["ingredients"].replace(np.nan, null)
csv["steps"] = csv["steps"].replace(np.nan, null)
csv["img"] = csv["img"].replace(np.nan, null)

for i in range(len(csv)):
    try:
        cursor.execute("INSERT INTO recipes VALUES(%s,%s,%s,%s,%s,%s)",
                    (csv["recipe_name"][i],csv["serves"][i],csv["cooking_time"][i],
                        csv["ingredients"][i],csv["steps"][i],csv["img"][i]))
        db_conn.commit()
    except:
        print(f'Step {i} failed')
        
cursor.close()