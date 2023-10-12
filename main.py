import os
import sys
sys.path.append("ferry_service_application")
from ferry_service_application import _1_generate_demand, _2_assign_stations, _3_clean_requests, _4_optimize_ferry__routes, config
import sqlite3 as sl
import pandas as pd

def csv_to_db(df, name_of_table):
    # write csv to DB
    print('....Result to DB.....')
    conn = sl.connect('FERRY_SERVICE_DB')
    df.to_sql(name_of_table, conn, index=False, if_exists='replace')
    conn.commit()
    conn.close()
    print('finished.')

def get_avg_time_savings_percent(ferries):
    conn = sl.connect('FERRY_SERVICE_DB')
    #sql_statement = "SELECT AVG(rel_savings_t) FROM instance_size_10_pax_new WHERE  fleet_size = ?", (k)
    sql_statement = 'SELECT AVG("{column}") FROM "{table}" WHERE {fleet_size} = {k}'.format(column="rel_savings_t",
                                                                                      table="instance_size_10_pax_new",
                                                                                      fleet_size="fleet_size", k = ferries)
    data = conn.execute(sql_statement)
    for row in data:
        try:
            return (round(row[0], 2))
        except:
            return (row)

def get_avg_time_savings_absolute(ferries):
    conn = sl.connect('FERRY_SERVICE_DB')
    #sql_statement = "SELECT AVG(rel_savings_t) FROM instance_size_10_pax_new WHERE  fleet_size = ?", (k)
    sql_statement = 'SELECT AVG("{column}") FROM "{table}" WHERE {fleet_size} = {k}'.format(column="net_savings_t",
                                                                                      table="instance_size_10_pax_new",
                                                                                      fleet_size="fleet_size", k = ferries)
    data = conn.execute(sql_statement)
    for row in data:
        try:
            return (round(row[0], 2))
        except:
            return (row)

def get_avg_delay(ferries):
    conn = sl.connect('FERRY_SERVICE_DB')
    #sql_statement = "SELECT AVG(rel_savings_t) FROM instance_size_10_pax_new WHERE  fleet_size = ?", (k)
    sql_statement = 'SELECT AVG("{column}") FROM "{table}" WHERE {fleet_size} = {k}'.format(column="DELAY",
                                                                                      table="instance_size_10_pax_new",
                                                                                      fleet_size="fleet_size", k = ferries)
    data = conn.execute(sql_statement)
    for row in data:
        try:
            return (round(row[0], 2))
        except:
            return (row)


def get_average_dist():
    conn = sl.connect('FERRY_SERVICE_DB')
    sql_statement = 'SELECT AVG("{column}") FROM "{table}" WHERE {fleet_size}'.format(column="rel_savings_d", table="instance_size_10_pax_new", fleet_size=10)
    data = conn.execute(sql_statement)
    for row in data:
        try:
            return (round(row[0], 2))
        except:
            return(row)
    #print(data.fetchone())


def prep():
    print("Step 1: **generating passenger requests...**")
    _1_generate_demand.run()
    print("... requests generated.")
    print("_______________________")
    print("Step 2: **assign passenger requests to ferry stations ...**")
    _2_assign_stations.run()
    print("... requests assigned.")
    print("_______________________")
    print("Step 3: **reject dispensable requests where conventional-route < ferry-route...**")
    _3_clean_requests.run()
    print("... requests cleaned.")
    print("__________PREP_DONE_____________")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # genau einmal:
    if os.path.exists(config.OPTIMIZATION_ITER):
        os.remove(config.OPTIMIZATION_ITER)

    # SETS PER INSTANCE SIZE: 5
    for i in range(1):
        prep()
        print("__________PREP_DONE_____________")
        _4_optimize_ferry__routes.main()




        # in _4_optimize_ferry__routes: iteriert k = 1,2,...,10 und schreibt zu jeder iteration 10 neue linien zu csv.

    df_evaluate = pd.read_csv(config.OPTIMIZATION_ITER)

    #csv_to_db(df_evaluate, "instance_size_10_pax_new")

    k_ = [i for i in range(1, config.K_FLEET_SIZE + 1)]

    evaluation = {}
    evaluation2 = {}
    evaluation3 = {}
    avg_savings_abs_t = []
    index_labels =['avg_rel_time_savings', 'avg_abs_time_savings', 'avg_delay' ]

    for k in k_:
        evaluation[k]  = [get_avg_time_savings_percent(ferries=k),
                          get_avg_time_savings_absolute(ferries=k),
                          get_avg_delay(ferries=k)]

        #avg_savings_abs_t.append({k: get_avg_time_savings_absolute(ferries=k)})

    df = pd.DataFrame(evaluation, index = index_labels)

    # styled_df = df.style
    # pd.display(styled_df)
    print(df.to_string())




    print(get_avg_time_savings_percent(ferries = 3))
    #get_average_dist()









    """
    Step 1: create a new table per set. (csv first, then Datatable)
    each set is optimized with k = 2,4,6,8,10 vehicles
    --> 5 iterations per set, 5 times adding 10 rows to the new csv. 
    
    
    
    
        Microspective: Ein Set besteht aus einer Instanz; Eine Instanz besteht aus 10 Pax-Requests
        
        
        
        
        
        Macrospective:  
            1.Ein Set besteht aus 10 Instanzen; eine Instanz besteht aus 10 Pax-Requests
            2.Ein Set besteht aus 10 Instanzen; eine Instanz besteht aus 20 Pax-Requests
            3.Ein Set besteht aus 10 Instanzen; eine Instanz besteht aus 30 Pax-Requests
            4.Ein Set besteht aus 10 Instanzen; eine Instanz besteht aus 40 Pax-Requests
    """

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
