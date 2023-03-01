# Satellite Scheduling System
# Written by Glyn Jones

# Allocates satellites to user areas in order of UA number.
# Allocates satellites by longest available to the UA
# No handover logic - assumes phased array satellite can manage transition between UAs.

# Expected Inputs (from Input_Files directory):

# SSS_Start_File.csv is in 2 parts:
# The top 2 lines provide the start UTC and end UTC for the scheduling period
# The following lines provide the allocation of satellites to UAs (and handover status) at the end of the last schedule
# from the previous SSS schedule.  The operator will have to add in any new or take away any removed UAs.
# it has the format (example):
#         utc_start_date            utc_end_date
#     2022 Jul 25 00:00:00      2022 Jul 25 23:59:00
#
#     ua_number   ua_name     prev_sat    curr_sat    ho_utc      ua_filename
#          0      London        S11         None       None       UA_London.csv

# UA_<name>.csv  Provides satellite rising and setting times for each User Area (UA) in a list.
# it comes from STK (or other software) as csv files.
# it has the format (example):
#       satCatlog    state            datetime               elev                  azmth                 distance
#       S11          rising   2022 Jul 26 13:46:22      65.66161554276742    344.49141587652036    19926.09269267054
# Where rising indicates S11 can provide coverage to the UA from that time onwards.

# Produces Outputs (in Output_Files directory):

# UAF_<name>.csv Aggregate of ua_data and ua_coverage in a df
# it has the format (example):
#       utc_off   ua_no      ua_name     prev_sat    cur_sat     ho_utc       S11       S12    S13 .....
#           0       1        London         S11        S12         5           0         1       0
#           0       2        Paris          S21        S21        None         0         1       0

# h/o utcs are not introduced as new UTC epochs - this could be added later.
# The prev_sat and cur_sat satellites allocated within a handover window need to be locked (aren't at the moment)
# All the satellite availability columns are the same as are input (may help reviewing the allocation algorithm)

import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

# Sets the display columns and width

pd.set_option("display.max_columns", 500)               # widens the column width so can display all satellites
pd.set_option('display.width', 500)                     # widens the display width so can display all satellites

# CONFIGURABLE PROGRAM PARAMETERS *************************************************************************************
# CONFIGURABLE PROGRAM PARAMETERS *************************************************************************************

# Configurable program parameters:

h = int(1)                                                      # Handover period in minutes
backfill_enabled = True                                         # Backfill of input sat data "True" or "False"

# FUNCTIONS ***********************************************************************************************************
# FUNCTIONS ***********************************************************************************************************


def get_position(list, item):

    # Get the position of an item in a list
    # The inputs are the list of items, and the item whose position is needed
    # The output gives its position of the item in the list (first position is zero) e.g. 3

    s = len(list)
    j = 0
    while j < s:
        if list[j] == item:
            break
        else:
            j += 1
    return j


def backfill(ms_df, counter, uaposition, satposition, state, n):

    # Backfills the df with the status of the satellite in all previous UTC epochs (utc_df) for that UA
    # so that the rise/set satellite transition data doesn't need to be provided for hours in advance
    # However, it must only backfill NaNs, not 1s or 0s so it doesn't overwrite previous statuses.
    # If the satellite is setting, then backfills with 1 (i.e. was previously available)
    # If the satellite has just risen, then backfills with 0 (i.e. was previously unavailable)
    # Returns ms_df

    while counter > 0:

        if ms_df.iat[((counter-1)*n) + uaposition, satposition] != "NaN":
            break

        else:
            if state == "setting":
                ms_df.iat[((counter-1)*n) + uaposition, satposition] = 1

            else:
                ms_df.iat[((counter-1)*n) + uaposition, satposition] = 0

        counter -= 1

    return ms_df


def count_length(df3):

    # Counts the number of utc epocs each satellite is available for
    # You'd think this would be possible using a simple pandas command but no it isn't so have to resort to this
    # It counts consecutive 1's for each satellite and returns the number in a df in the 'score' column
    # df4 has its rows and columns transposed from df3 to make it easier to sort in the calling program

    # Sets up variables

    j = 0                                                   # Counts along columns of df3 (Satellites)
    max_rows = len(df3)
    max_cols = len(df3.columns)

    df3 = df3.copy(deep=True)
    df4 = df3.transpose(copy=True)                          # Transposes df4 rows and columns from df3
    df4.rename(columns={0: "score"}, inplace=True)
    df4["score"] = 0                                        # Sets the score column to zero

    # Loop to count to the first zero and populate the result into df4

    while j < max_cols:
        i = 0                                               # Counts down rows of df3
        while i < max_rows:
            x = df3.iat[i, j]
            if x == 1:
                df4.iat[j, 0] = df4.iat[j, 0] + 1   # Adds 1 to the score.
                i = i + 1
            else:
                break
        j = j + 1

    return df4


def availsats(i, s, utc_df, sat_names):

    # Provides a list of all available satellites to a user area
    # It works as follows:
    # One by one identifies the satellites with a '1' in the UA row. Gets their names. Adds them to the list
    # Inputs:
    # i = the user area number
    # s = the number of satellites
    # utc_df = the dataframe for the epoch
    # sat_names = the list of satellite names
    # Outputs:
    # sat_list = names of each available satellite by user area

    # Take the sat_list

    j = 0                                           # counts columns (i.e. satellites)
    avail_sat_list = []                             # starts with a blank satellite list

    while j < s:
        x = utc_df.iat[i, 7+j]
        if x == 1:
            sat = sat_names[j]
            avail_sat_list.append(sat)
        j = j + 1

    return avail_sat_list


def lookahead(i, n, counter, df, avail_unalloc_sats):

    # Allocates a new satellite to a UA from a list of available unallocated satellites
    # It takes the full df and looks forward in it to see which satellite is available to the UA the longest

    # Inputs:
    # i - the UA number. This could be for example 0 which might correspond to London.
    # n - the total number of UAs. This might be for example 5 or 10 or 50.
    # utc_off - the utc offset from which the look-ahead in the df will start
    # df - the full df from the main program and is not modified in any way.
    # avail_unallocated_sats - the list of names of satellites in the form ["S41", "S42", "S43", "S44"]

    # Outputs:
    # sat (the name of the satellite to be used e.g. S42)

    # Creates the list of available unallocated satellite names for the UA

    columns = df.columns                                # Gets the df column headings
    column_index = columns.get_indexer(avail_unalloc_sats)  # Matches them against avail_unalloc to get column nos
    df3 = df.iloc[counter:, column_index]               # Selects just the available unallocated sat columns in the df
    df3 = df3.copy()                                    # Makes df3 not just a slice of df but in its own memory
    df3 = df3.iloc[i::n]                                # Then selects just the UA of interest from the df
    df3 = df3.reset_index(drop=True)                    # Gets rid of the index (which is all the same)

    # Calls the function to calculate how long the satellites are available for

    count_to_first_zero = count_length(df3)             # Finds out how long satellites are available

    # Manipulates the output to find out which satellite is available for the longest & chooses it

    w = count_to_first_zero.sort_values("score", ascending=False)  # Sorts the list in descending order
    w = pd.DataFrame(w)                                 # To avoid wierd intermittent python error
    sat = w.index.values[0]                             # Presents the sat name at index zero (the longest run).

    return sat


def allocatesat(s, n, counter, ms_df, sat_names, h):

    # Satellite allocation algorithm
    # This ignores handovers for the time being - it may all change if we move baseline
    # Takes the inputs and decides which satellite to allocate to each user area
    # Does it sequentially so that 1st user area gets best availability, then second, then third etc.
    # This way you know which user areas you can provide service to and which you need to drop
    # Because some user areas will be closer to others, the output availability may not be be linear!
    # Inputs:
    # s = the number of satellites e.g. 16
    # n = the number of user areas e.g. 4
    # counter = the counter to give the correct utc epoch
    # ms_df = the full dataframe for all epochs
    # sat_names = the list of satellites names in the form ["S11", "S12", "S13", "S14".......]
    # h = the handover time
    # Outputs:
    # "cur_sat" and "ho_utc" columns in the utc_df

    utc_df = ms_df.loc[counter:counter+(n-1), :].copy()         # Create working df for the relevant utc_off epoch
    utc_df = utc_df.reset_index(drop=True)

    # Repeat this loop incrementing i (user area) until all UAs are allocated a satellite or an ERR

    i = int(0)  # UA counter

    while i < n:                                                   # Loop through all UAs sequentially

        # Find out what satellites are available to the UA

        avail_sat_list = availsats(i, s, utc_df, sat_names)

        # Is prev_sat available?

        if utc_df.at[i, "prev_sat"] in avail_sat_list:

            # If so, allocate it.

            utc_df.at[i, "cur_sat"] = utc_df.at[i, "prev_sat"]
            utc_df.at[i, "ho_utc"] = "None"

            # Then remove prev_sat (= cur_sat) from all UAs in the epoch

            cur_sat = utc_df.loc[i, "cur_sat"]
            cur_sat_position = get_position(utc_df.columns, cur_sat)
            utc_df[utc_df.columns[cur_sat_position]] = 0                # Removes the allocated sat from all UAs
            utc_df.iat[i, cur_sat_position] = 1                 # Restores the allocated satellite to the chosen UA

        else:
            # If prev_sat isn't available.....

            # If 0 satellites are available, raise an error message and mark curr_sat as "ERR"

            if len(avail_sat_list) == 0:
                utc_df.at[i, "cur_sat"] = "ERR"                     # Flags an error message if no satellites available

                if utc_df.at[i, "prev_sat"] == "ERR":               # Is the prev_sat an ERR?
                    utc_df.at[i, "ho_utc"] = "None"                 # If so then utc_off = None

            else:

                # If some satellites are available, find the longest

                longest_sat = lookahead(i, n, counter, ms_df, avail_sat_list)  # Find the longest lasting sat

                # Selects the longest sat

                utc_df.at[i, "cur_sat"] = longest_sat               # Populates cur_sat
                columns = utc_df.columns

                # Determine the index position of longest sat

                column_index = get_position(columns, longest_sat)

                utc_df[utc_df.columns[column_index]] = 0            # Removes the allocated satellite from all UAs
                utc_df.iat[i, column_index] = 1                     # Restores the allocated satellite to the chosen UA

        i = i + 1

    update2 = utc_df.loc[:, ["cur_sat", "ho_utc"]]              # Take "cur_sat" & "ho_sat" columns
    update2.index = update2.index + counter                     # Set the correct index to match the df
    ms_df.update(update2)                                      # Update df with them

    return ms_df


def create_blank_UTC_df(start_data, ua_df, n):

    # This creates a blank UTC df based on the no. of user areas and no of satellites
    # It consists of a utc_offset, ua info, then all the satellite names as columns
    # All values should be 0, NONE or NA apart from UA names, US numbers and prev_sat
    # headings = "utc_off", "ua_name", "avail_sats", "S11", "S12", "S13"......
    # n = no. of UAs
    # Outputs:
    # ua_df = the UA dataframe
    # s = no. of satellites
    # uas = list of UA names
    # sat_names = the list of names of all satellites

    sat_names = pd.unique(ua_df["satCatlog"])
    sat_names = np.sort(sat_names)
    s = len(sat_names)
    uas = start_data.loc[:, "ua_name"]

    # Create the UTC df. This is produced at every UTC time there is a satellite transition

    array = np.full([n, s], "NaN")
    sat_df = pd.DataFrame(array, columns=sat_names)
    start_data.drop("ua_filename", inplace=True, axis=1)
    start_data.insert(0, "utc_off", 0)
    start_data.insert(6, "avail_sats", 0)
    utc_df = pd.concat([start_data, sat_df], axis=1)
    utc_df["prev_sat"] = utc_df["cur_sat"]
    utc_df["cur_sat"] = "None"

    return utc_df, s, uas, sat_names


# IMPORT SECTION ******************************************************************************************************
# IMPORT SECTION ******************************************************************************************************

# Imports the SSS start file, which contains the start UTC, the end UTC, (in the top 2 lines), the
# UA names, previous satellite allocations at the end of the last schedule, the UA csv file names for each UA and
# the gateway antenna names together with the gateway antenna in use.
# Imports the UA files with satellite rising and setting times
# Imports the frequency co-ordination file (just a placeholder to allow UAFs to be produced)

utc = pd.read_csv("../SSS/Input_Files/SSS_Start_File.csv", nrows=1)
start_data = pd.read_csv("../SSS/Input_Files/SSS_Start_File.csv", skiprows=2)
freq_coord = pd.read_csv("../SSS/Input_Files/SSS_Freq_Coord_File.csv")

# Program variables

n = len(start_data)
i = int(0)
seq_no = 1
ua_df = pd.DataFrame()
utc_start_date_s = str(utc.at[0, "utc_start"])
utc_start_date = datetime.strptime(utc_start_date_s, "%Y %b %d %H:%M:%S")
utc_start = int(0)
utc_end_date = str(utc.at[0, "utc_end"])
utc_end_date = datetime.strptime(utc_end_date, "%Y %b %d %H:%M:%S")
utc_end = int((utc_end_date - utc_start_date).total_seconds() / 60)
now = datetime.now()
dt_runtime = now.strftime("%d/%m/%Y %H:%M:%S")

gw_df = start_data.loc[:, ("cur_gw", "other_gw")]
gw_df["ua_no"] = start_data["ua_no"]
gw_df = gw_df.loc[:, ["ua_no", "cur_gw", "other_gw"]]
start_data.drop(["cur_gw", "other_gw"], inplace=True, axis=1)

# Loop to import the UA filenames listed in ua_start_data, add a UA name column and aggregate them.
# UA files are generated from STK (or another source)
# They are a list of satellite rising and setting times for each UA with additional fields which are unused.

while i < n:
    ua_filename = start_data.loc[i, "ua_filename"]
    ua_filename = "Input_Files/" + ua_filename
    ua_coverage = pd.read_csv(ua_filename)
    ua_name = start_data.loc[i, "ua_name"]
    ua_coverage.insert(1, "UA", ua_name)

    ua_df = pd.concat([ua_df, ua_coverage])

    i += 1

# PROCESSES THE INPUT DATA *******************************************************************************************
# PROCESSES THE INPUT DATA *******************************************************************************************

# Renumber the ua_df index

ua_df = ua_df.reset_index(drop=True)

# Delete the unnecessary columns in the df (elevation, azimuth, and distance)

ua_df.drop("elev", inplace=True, axis=1)
ua_df.drop("azmth", inplace=True, axis=1)
ua_df.drop("distance", inplace=True, axis=1)

# Delete the unnecessary rows (culminate) and reset the index to start at zero

ua_df = ua_df.loc[ua_df["state"] != "culminate"].copy()
ua_df.reset_index(drop=True, inplace=True)

# Define the variables
# ua_df_counter = counts down the rows of ua_df i.e. the satellite transitions
# ua_df_counter_end = total no. of satellite transitions ( = the number of utc epochs)

ua_df_counter = 0
ua_df_counter_end = len(ua_df)

# Calculate the UTC offset down the df replacing the datetime with its UTC offset.

while ua_df_counter < ua_df_counter_end:

    date_time = str(ua_df.loc[ua_df_counter, "datetime"])
    utc_off = datetime.strptime(date_time, "%Y %b %d %H:%M:%S")
    utc_off = utc_off - utc_start_date
    utc_off = int(utc_off.total_seconds() / 60)
    ua_df.iat[ua_df_counter, 3] = utc_off

    ua_df_counter += 1

ua_df = ua_df.rename(columns={'datetime': 'utc_off'})

# Sort the ua_df by utc_offset and re-index

ua_df = ua_df.sort_values("utc_off")
ua_df.reset_index(drop=True, inplace=True)

# Create the start UTC df

utc_df, s, uas, sat_names = create_blank_UTC_df(start_data, ua_df, n)

# DISPLAYS THE INTRODUCTION MESSAGE AND SOFTWARE REVISION *************************************************************
# DISPLAYS THE INTRODUCTION MESSAGE AND SOFTWARE REVISION *************************************************************

print("\n")
print("\033[1m" + "METHERA SATELLITE SCHEDULING SYSTEM" + "\033[0m")
print("Version 75. Feb 2023.  by Glyn. \n \n")
print("Analysing Inputs:")

# BUILD THE MASTER SCHEDULE FROM THE START OF THE SCHEDULING PERIOD TO THE END OF THE SCHEDULING PERIOD ***************
# BUILD THE MASTER SCHEDULE FROM THE START OF THE SCHEDULING PERIOD TO THE END OF THE SCHEDULING PERIOD ***************

# For each UTC determine the availability of each satellite by building up a complete table over time.
# The data will be incomplete for several hours......

ms_df = utc_df.copy()

utc_counter: int = 0

# Loop to populate the ms_df

for utc_counter in tqdm(range(utc_end)):

    # Populate the UTC for the ms_df

    var1 = utc_counter * n
    var2 = var1 + n - 1
    ms_df.loc[var1:var2, "utc_off"] = utc_counter

    # Identify whether there are any changes to satellite availability at this UTC & count the number

    change_df = ua_df.loc[ua_df["utc_off"] == utc_counter]

    change_df_count = len(change_df)
    change_df = change_df.reset_index(drop=True)

    while change_df_count > 0:

        # Take the rising or setting data at UTC and populate the UA satellite status as 1 or 0

        satCatlog = change_df.at[(change_df_count-1), "satCatlog"]
        satposition = get_position(ms_df.columns, satCatlog)
        state = change_df.at[(change_df_count-1), "state"]
        ua = change_df.at[(change_df_count-1), "UA"]
        uaposition = get_position(uas, ua)

        if state == "rising":
            ms_df.iat[(uaposition+(utc_counter * n)), satposition] = 1            # Set sat to 1 to show available

        if state == "setting":
            ms_df.iat[(uaposition+(utc_counter * n)), satposition] = 0            # Set sat to 0 to show not available

        if backfill_enabled:

            if utc_counter > 0:
                ms_df = backfill(ms_df, utc_counter, uaposition, satposition, state, n)

        change_df_count -= 1

    temp_df = ms_df.loc[ms_df["utc_off"] == utc_counter]
    ms_df = pd.concat([ms_df, temp_df], axis=0, join="outer")
    ms_df = ms_df.reset_index(drop=True)
    utc_counter += 1

    # Populate the ms_df 'available sats' column

ms_df['avail_sats'] = ms_df.iloc[:, 7:(7+s)].sum(axis=1)
ms_df = ms_df.astype({"avail_sats": int})

# MAIN PROGRAM  *******************************************************************************************************
# MAIN PROGRAM  *******************************************************************************************************

# program variables:

ms_df_counter = int(0)                                          # Main program counter (counts UTC offsets)
ms_df_counter_end = len(ms_df)                                  # Number of UA lines from start to finish
utc_off = int(0)                                                # Offset starts at zero from utc
s = len(sat_names)                                              # No. of satellites

# Variables for the analytics:

max_sats = int(0)
sats_used_total = int(0)                                        # Total number of satellites used (for analytics)

# Displays the key parameters from the input file:

print("\n")
print("Satellites: ", s)
print("User Areas:", n)
print("Handovers are assumed instantaneous (not modelled)")
print("All gateways: in-region")
print("File Sequence Number: 1")
fileSeqNo = 1
print("Backfill of satellite status to start UTC time:", backfill_enabled, "\n")
print("Start UTC (yyyy-mm-dd):", utc_start_date)
print("End UTC (yyyy-mm-dd):", utc_end_date)

# ANALYTICS SECTION FOR THE INPUT DATA *******************************************************************************
# ANALYTICS SECTION FOR THE INPUT DATA *******************************************************************************

# Schedule Duration Analysis

total_time = utc_end - utc_start
total_satellite_minutes = total_time * s
total_ua_minutes = total_time * n

# Satellite Availability Analysis

avail_sats_distn = ms_df.loc[:, "avail_sats"]
avail_sats_distn = avail_sats_distn.value_counts(ascending=True).sort_index()

print("Schedule duration (hours / minutes): ", int(total_time/60), "hours", (total_time % 60), "minutes")
print("Schedule duration (minutes):", total_time, "\n")

print("User area minutes (no. user areas x schedule duration):", total_ua_minutes)
print("Available satellite minutes (no. satellites x schedule duration):", total_satellite_minutes)
print("Target Utilisation  (User area minutes / Available satellite minutes):",
      int(100*(total_ua_minutes/total_satellite_minutes)), "%\n")

print("The lowest number of satellite beams available MUST BE GREATER THAN the\n"
      "number of user areas within the visible earth to avoid outages!")
print("The number of satellite beams over user areas (no. of occurrences):")
print(avail_sats_distn.to_string())
print("\n")
print("Calculating Schedule:")

# MAIN PROGRAM LOOP FOR EACH UTC EPOCH ALLOCATING THE SATELLITES ******************************************************
# MAIN PROGRAM LOOP FOR EACH UTC EPOCH ALLOCATING THE SATELLITES ******************************************************

for ms_df_counter in tqdm(range(0, ms_df_counter_end, n)):

    # Invokes the satellite allocation function

    ms_df = allocatesat(s, n, ms_df_counter, ms_df, sat_names, h)

    # Updates the df for the next UTC epoch and the next loop:

    update = ms_df.loc[ms_df_counter:ms_df_counter+(n-1), "cur_sat"]  # Get the "cur_sat" column for the utc
    update = update.to_frame(name="prev_sat")                   # Form it into a DataFrame
    update.index = update.index + n                             # Set the index to match the df
    ms_df.update(update)                                        # Update df prev_sat with the previous utc cur_sat

# Prints the Master Schedule on screen

print("\nMASTER SCHEDULE:\n", ms_df)

# ANALYTICS SECTION FOR THE OUTPUT SCHEDULE ***************************************************************************
# ANALYTICS SECTION FOR THE OUTPUT SCHEDULE ***************************************************************************

# Handover Analysis

handovers = ms_df[ms_df["cur_sat"] != ms_df["prev_sat"]]        # df of handovers
no_handovers = len(handovers)                                   # No of handovers
handover_distn = handovers[["ua_no", "ua_name"]]
d3 = pd.handover_distn.value_counts(ascending=True)
d3 = pd.DataFrame(d3)
time_in_handover = (0)

# Outage Analysis bwiehwilhewoih



i = 0                                                           # UA counter
ms_df_counter = 0                                               # Set counter to zero
ua_outages = pd.DataFrame(uas)                                  # get the UA names
ua_outages["outage"] = 0                                        # Set the outage column to zero

while ms_df_counter < ms_df_counter_end:

    if ms_df.loc[ms_df_counter, "cur_sat"] == "ERR":
        index = ms_df_counter % n
        ua_outages.loc[index, "outage"] = ua_outages.loc[index, "outage"] + 1
    ms_df_counter += 1

outage_total = ua_outages.loc[:, "outage"].sum(axis=0)
total_availability = int(100 * (1 - (outage_total / total_ua_minutes)))

# Satellite Usage Analysis

av_sats_used = round((((total_time * n) - outage_total) / total_time) + (time_in_handover/total_time), 2)

# Combined Outages and Handovers

combined_o_ho = pd.concat(ua_outages, handover_distn)

# Prints the analytics

print("Handovers:")
print("Number of handovers:", no_handovers)
print("Time performing handovers (minutes):", time_in_handover)
print("% of available satellite minutes used in handovers:",
      round((time_in_handover/total_satellite_minutes)*100, 2), "%\n")
print("handover distribution", handover_distn)

print("Satellite Usage:")
print("Average satellites concurrently used:", av_sats_used)
print("Average satellites concurrently unused:", round((s - av_sats_used), 2), "\n")

print("Outages:")
print("User area outage time (minutes):", outage_total)
print("Average availability across all user areas:", round(total_availability, 2), "%\n")
print("Outage time by user area (minutes):")
print(ua_outages)

# WRITES OUTPUTS TO FILES *********************************************************************************************
# WRITES OUTPUTS TO FILES *********************************************************************************************

# Writes the Master Schedule to .csv

ms_df.to_csv("Output_Files/SSS_Master_Schedule_" + utc_start_date_s + "_" + str(seq_no) + ".csv")

quit()

# Creates a UAF and GAF file for each UA, writes it as a json file with the activation date/time and unique ref

ua_counter = 0

while ua_counter < n:

    # Creates the UAF file

    uaf = ms_df.iloc[:, 0:6]
    uaf = (uaf.loc[ua_counter::n, :])
    uaf.drop("ua_no", inplace=True, axis=1)
    uaf = uaf.loc[uaf["ho_utc"] != "None"]
    uaf = uaf[(uaf['prev_sat'] != "ERR") | (uaf['cur_sat'] != "ERR")]
    uaf.reset_index(drop=True, inplace=True)
    uaf_name = uaf.iloc[0, 1]
    uaf["time1"] = pd.to_timedelta(uaf["utc_off"], "m")
    uaf["UTC"] = utc_start_date + uaf["time1"]
    uaf.drop(["utc_off", "time1", "ua_name", "ho_utc", "prev_sat"], inplace=True, axis=1)
    uaf = uaf[["UTC", "cur_sat"]]
    uaf = uaf.rename(columns={'UTC': 'startTime', 'cur_sat': 'SatCatNo'})

    uaf["altAcquisitionFreq"] = freq_coord.loc[ua_counter, "altAcquisitionFreq"]
    uaf["polarisation"] = freq_coord.loc[ua_counter, "polarisation"]
    uaf["symbolRate"] = freq_coord.loc[ua_counter, "symbolRate"]
    uaf["endTime"] = uaf["startTime"].shift(-1)
    uaf = uaf.reindex(columns=['SatCatNo', 'startTime', 'endTime', 'altAcquisitionFreq', 'polarisation', 'symbolRate'])
    uaf_j = uaf.to_json(orient="index")
    f = open("Output_Files/UAF_" + uaf_name + "_" + utc_start_date_s + "_" + str(fileSeqNo), "w")
    f.write(uaf_j)
    f.close()

    # Creates the GAF file

    gaf = uaf
    gaf.drop(["altAcquisitionFreq", "polarisation", "symbolRate"], inplace=True, axis=1)
    gaf.loc[::2, "gw_ant"] = gw_df.loc[ua_counter, "cur_gw"]
    gaf.loc[1::2, "gw_ant"] = gw_df.loc[ua_counter, "other_gw"]
    gaf_name = uaf_name
    gaf_j = gaf.to_json(orient="index")
    f = open("Output_Files/GAF_" + gaf_name + "_" + utc_start_date_s + "_" + str(fileSeqNo), "w")
    f.write(gaf_j)
    f.close()

    ua_counter += 1

print("\n")
print("Master schedule ", n, "UAF files", "and ", n, "GAF files written to file")

# SWAP PROCESS  *******************************************************************************************************
# SWAP PROCESS  *******************************************************************************************************

print("user area outages", ua_outages)

while i < n:
    outage_time = ua_outages.iloc[i, "outage"]
#    if outage_time != 0:








quit()
