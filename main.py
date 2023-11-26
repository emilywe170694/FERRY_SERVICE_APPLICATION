import os
import sys
import datetime

from matplotlib import pyplot as plt

sys.path.append("ferry_service_application")
from ferry_service_application import _1_generate_demand, _2_assign_stations, _3_clean_requests, \
    config
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

def get_avg_time_savings_percent(ferries, table_name, objective):
    conn = sl.connect('FERRY_SERVICE_DB')
    #sql_statement = "SELECT AVG(rel_savings_t) FROM instance_size_10_pax_new WHERE  fleet_size = ?", (k)
    sql_statement = 'SELECT AVG("{column}") FROM "{table}" WHERE {fleet_size} = {k} AND {objective} = "{obj}"'.format(column="rel_savings_t",
                                                                                      table= table_name,
                                                                                      fleet_size="fleet_size", k = ferries, objective="objective", obj = objective)
    data = conn.execute(sql_statement)
    for row in data:
        try:
            return (round(row[0], 2))
        except:
            return (row)

def get_avg_time_savings_minutes(ferries, table_name, objective):
    conn = sl.connect('FERRY_SERVICE_DB')
    #sql_statement = "SELECT AVG(rel_savings_t) FROM instance_size_10_pax_new WHERE  fleet_size = ?", (k)
    sql_statement = 'SELECT AVG("{column}") FROM "{table}" WHERE {fleet_size} = {k} AND {objective} = "{obj}"'.format(
        column="net_savings_t",
        table=table_name,
        fleet_size="fleet_size", k=ferries, objective="objective", obj=objective)
    data = conn.execute(sql_statement)
    for row in data:
        try:
            return (round(row[0], 2))
        except:
            return (row)

def get_avg_delay(ferries, table_name, objective):
    conn = sl.connect('FERRY_SERVICE_DB')
    #sql_statement = "SELECT AVG(rel_savings_t) FROM instance_size_10_pax_new WHERE  fleet_size = ?", (k)
    sql_statement = 'SELECT AVG("{column}") FROM "{table}" WHERE {fleet_size} = {k} AND {objective} = "{obj}"'.format(column="DELAY",
                                                                                      table=table_name,
                                                                                      fleet_size="fleet_size", k = ferries, objective="objective", obj=objective)
    data = conn.execute(sql_statement)
    for row in data:
        try:
            return (round(row[0], 2))
        except:
            return (row)


def get_net_dist_savings_pax(table_name):
    conn = sl.connect('FERRY_SERVICE_DB')
    sql_statement = 'SELECT AVG("{column}") FROM "{table}" WHERE {fleet_size}'.format(column="net_savings_d", table=table_name, fleet_size=10)
    data = conn.execute(sql_statement)
    for row in data:
        try:
            return (round(row[0], 2))
        except:
            return(row)

def get_rel_dist_savings_pax(table_name):
        conn = sl.connect('FERRY_SERVICE_DB')
        sql_statement = 'SELECT AVG("{column}") FROM "{table}" WHERE {fleet_size}'.format(column="rel_savings_d",
                                                                                          table=table_name,
                                                                                          fleet_size=10)
        data = conn.execute(sql_statement)
        for row in data:
            try:
                return (round(row[0], 2))
            except:
                return (row)
    #print(data.fetchone())

def get_avg_distance_traveled(ferries, table_name, objective):
    conn = sl.connect('FERRY_SERVICE_DB')
    sql_statement = 'SELECT AVG("{column}") FROM "{table}" WHERE {fleet_size} = {k} AND {objective} = "{obj}"'.format(column="distance_traveled",
                                                                                      table=table_name,
                                                                                      fleet_size="fleet_size", k = ferries, objective="objective", obj=objective)
    data = conn.execute(sql_statement)
    for row in data:
        try:
            return (round(row[0], 2))
        except:
            return (row)


def get_sum_distance_traveled(ferries, table_name, objective):
    conn = sl.connect('FERRY_SERVICE_DB')
    #sql_statement = "SELECT AVG(rel_savings_t) FROM instance_size_10_pax_new WHERE  fleet_size = ?", (k)
    sql_statement = 'SELECT SUM("{column}") FROM "{table}" WHERE {fleet_size} = {k} AND {objective} = "{obj}"'.format(column="distance_traveled",
                                                                                      table=table_name,
                                                                                      fleet_size="fleet_size", k = ferries, objective="objective", obj=objective)
    data = conn.execute(sql_statement)
    for row in data:
        try:
            return (round(row[0], 2))
        except:
            return (row)


def assess(k, table_name):
    result = [get_avg_time_savings_percent(ferries=k, table_name=table_name),
              get_avg_time_savings_minutes(ferries=k, table_name=table_name),
              get_avg_delay(ferries=k, table_name=table_name)]
    return result


def generate_requests_for_optimization():
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


def query_savings_df(K_i, table_name, objective):
    time_savings_percent = []
    time_savings_minutes = []
    delay = []
    for k in K_i:
        time_savings_percent.append(get_avg_time_savings_percent(ferries=k, table_name=table_name, objective=objective))
        time_savings_minutes.append(get_avg_time_savings_minutes(ferries=k, table_name=table_name, objective=objective))
        delay.append(get_avg_delay(ferries=k, table_name=table_name, objective=objective))
    savings_df = pd.DataFrame({f'avg_time_savings_in_percent_{objective}': time_savings_percent,
                               f'avg_time_savings_in_minutes_{objective}': time_savings_minutes,
                               f'avg_delay_in_minutes_{objective}': delay}, index=K_i)
    return savings_df


def query_ferry_stats_df(K_i, table_name, objective):
    distance_traveled = []
    sum_distance_traveled = []
    for k in K_i:
        distance_traveled.append(get_avg_distance_traveled(ferries=k, table_name=table_name, objective=objective))
        sum_distance_traveled.append(get_sum_distance_traveled(ferries=k, table_name=table_name, objective=objective) / config.NUMBER_OF_INSTANCES)
    savings_df = pd.DataFrame({f'avg_distance_traveled_{objective}': distance_traveled,
                               f'sum_distance_traveled_{objective}': sum_distance_traveled}, index=K_i)
    return savings_df

def query_distance(table_name):
    rel_distance = get_rel_dist_savings_pax(table_name)
    net_distance = get_net_dist_savings_pax(table_name)
    for k in K_i:
        savings_df = pd.DataFrame({'distance_savings_in_percentage': rel_distance,
                               'distance_savings_in_meter': net_distance}, index=K_i)
    return savings_df


if __name__ == '__main__':
    #
    # K_i = [i for i in range(5, 10 + 1) if i % 2 == 0]  # K_i = [2,4,6,8,10]
    # K_i.insert(0, 1)
    # K_i.append(15)
    # K_i.append(config.K_FLEET_SIZE)
    K_i = [1,2,4,6,8,10]

    mode = ['operator','passenger']
    #
    timestamp = datetime.datetime.now()
    # # # #SETS PER INSTANCE SIZE: 5
    # if os.path.exists(config.EVALUATION_PAX):
    #     os.remove(config.EVALUATION_PAX)
    # if os.path.exists(config.EVALUATION_FERRY):
    #     os.remove(config.EVALUATION_FERRY)
    #
    # for instance in range(config.NUMBER_OF_INSTANCES):
    #     print('INSTANCE #', instance)
    #
    #     # new file required for each new instance, otherwise the data of the new instance will be appended to the old file (mode="a")
    #
    #     # Generates 1 instance of n passengers to optimize:
    #     generate_requests_for_optimization()
    #
    #     print("__________PREP_DONE_____________")
    #     from ferry_service_application import _4_optimize_ferry_routes
    #
    #     # Optimize routes for every k in K = [1,2,4,6,8,10]:
    #     _4_optimize_ferry_routes.run_with_iterations(K_i, mode)
    #
    #     # Write results for every instance to database FERRY_SERVICE_DB
    # df_evaluate = pd.read_csv(config.EVALUATION_PAX)
    # table_name = f"pax__instance_{instance}_size_{config.n_NUMBER_OF_PASSENGERS}_{timestamp}"
    # csv_to_db(df_evaluate, table_name)
    #
    # df_evaluate_K = pd.read_csv(config.EVALUATION_FERRY)
    # table_name_K = f"ferry__instance_{instance}_size_{config.n_NUMBER_OF_PASSENGERS}_{timestamp}"
    # csv_to_db(df_evaluate_K, table_name_K)

    table_name="pax__instance_2_size_10_2023-11-19 23:23:25.532857"
    table_name_K="ferry__instance_2_size_10_2023-11-19 23:23:25.532857"


    savings_operator_mode = query_savings_df(K_i, table_name, "operator")
    savings_passenger_mode = query_savings_df(K_i, table_name, "passenger")
    #
    ferry_stats_operator_mode = query_ferry_stats_df(K_i, table_name_K, "operator")
    ferry_stats_passenger_mode = query_ferry_stats_df(K_i, table_name_K, "passenger")
    #
    distance_savings = query_distance(table_name)
    #
    TOTAL = pd.concat([savings_operator_mode, savings_passenger_mode, ferry_stats_operator_mode, ferry_stats_passenger_mode, distance_savings], axis=1)
    TOTAL.to_csv(f'{config.PENALTY_FACTOR}_multiinstance_{timestamp}.csv', index=K_i)

    # Liniendiagramm
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Erster Plot (oben) für %-Savings
    axes[0, 0].plot(savings_operator_mode.index, savings_operator_mode['avg_time_savings_in_percent_operator'],
                    label='time savings in % (operator mode)', color='green')
    axes[0, 0].plot(savings_passenger_mode.index, savings_passenger_mode['avg_time_savings_in_percent_passenger'],
                    label='time savings in % (passenger mode)', color='blue')
    axes[0, 0].set_xticks(K_i)
    axes[0, 0].set_xlabel('Number of ferries')
    axes[0, 0].set_ylabel('Time Savings in %')
    axes[0, 0].set_title(f'Ferry Service with n = {config.n_NUMBER_OF_PASSENGERS} Passenger requests')
    axes[0, 0].legend()

    axes[0, 1].plot(savings_operator_mode.index, savings_operator_mode['avg_time_savings_in_minutes_operator'],
                    label='time savings in min (operator-mode)', color='green')
    axes[0, 1].plot(savings_passenger_mode.index, savings_passenger_mode['avg_time_savings_in_minutes_passenger'],
                    label='time savings in min (passenger-mode)', color='blue')
    axes[0, 1].set_xticks(K_i)
    axes[0, 1].set_xlabel('Number of ferries')
    axes[0, 1].set_ylabel('Time Savings in Minutes')
    axes[0, 1].set_title(f'Ferry Service with n = {config.n_NUMBER_OF_PASSENGERS} Passenger requests')
    axes[0, 1].legend()

    axes[1, 0].plot(savings_operator_mode.index, savings_operator_mode['avg_delay_in_minutes_operator'],
                    label='average delay (operator-mode)', color='green')
    axes[1, 0].plot(savings_passenger_mode.index, savings_passenger_mode['avg_delay_in_minutes_passenger'],
                    label='average delay (passenger-mode)', color='blue')
    axes[1, 0].set_xticks(K_i)
    axes[1, 0].set_xlabel('Number of ferries')
    axes[1, 0].set_ylabel('Time Savings in Minutes')
    axes[1, 0].set_title(f'Ferry Service with n = {config.n_NUMBER_OF_PASSENGERS} Passenger requests')
    axes[1, 0].legend()

    axes[1, 1].plot(ferry_stats_operator_mode.index, ferry_stats_operator_mode['avg_distance_traveled_operator'],
                    label='distance per ferry (operator-mode)', color='green', linestyle='--')
    axes[1, 1].plot(ferry_stats_passenger_mode.index, ferry_stats_passenger_mode['avg_distance_traveled_passenger'],
                    label='distance per ferry (passenger-mode)', color='blue', linestyle='--')

    axes[1, 1].plot(ferry_stats_operator_mode.index, ferry_stats_operator_mode['sum_distance_traveled_operator'],
                    label='total distance (operator-mode)', color='green')
    axes[1, 1].plot(ferry_stats_passenger_mode.index, ferry_stats_passenger_mode['sum_distance_traveled_passenger'],
                    label='total distance (passenger-mode)', color='blue')
    axes[1, 1].set_xticks(K_i)
    axes[1, 1].set_xlabel('Number of ferries')
    axes[1, 1].set_ylabel('Distance in meters ')
    axes[1, 1].set_title(f'Ferry Service with n = {config.n_NUMBER_OF_PASSENGERS} Passenger requests')
    axes[1, 1].legend()

    #plt.savefig(f'PLOT_{timestamp}.png')

    # Anzeigen der Diagramme
    plt.tight_layout()
    plt.show()


    """
   # FÜR dazugehörigen dataframe!!!
    df = pd.DataFrame(data)

    # Erstellen Sie ein Figure-Objekt und Subplots mit 1 Zeile und 2 Spalten
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))  # 1 Zeile und 2 Spalten

    # Plot in der ersten Spalte (ax1)
    ax1.plot(df['X-Achse'], df['avg_time_savings_in_percent'], label='Zeiteinsparungen in Prozent')
    ax1.set_xlabel('X-Achse')
    ax1.set_ylabel('Durchschnittliche relative Zeiteinsparungen')
    ax1.legend()

    # DataTable in der zweiten Spalte (ax2)
    ax2.axis('off')  # Versteckt die Achsen
    ax2.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

    plt.show()

    # after every iteration:
  







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
