# import sys
#
# sys.path.append("ferry_service")
#
# import ferry_service.path as ferry_app
# import ferry_service._1_generate_demand
# import ferry_service._2_assign_stations
# import ferry_service._3_clean_requests
# import ferry_service._4_optimize_ferry__routes
#
# import pandas as pd
#
#
#
# n                   = 6
# departure_window    = 30
# max_waiting         = 4
# service_time        = 1
#
#
# def set_parameters():
#     params = [
#         ("n", n),
#         ("departure", departure_window),
#         ("w_t", max_waiting),
#         ("dt", service_time)
#     ]
#     param_df = pd.DataFrame(params, columns=['param', 'value'])
#     param_df.to_csv(ferry_app.PARAMETER_SETTINGS, index=False)
#
#
