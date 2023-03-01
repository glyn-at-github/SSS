sat_names = "S11", "S12", "S13", "S14", "S15"
prev_sat_list = ["S14", "S15", "S17", "S18"]
sat_list = []
x = 0

while x < 5:
    sat = sat_names[x]
    if sat not in prev_sat_list:
        sat_list.append(sat)
    x = x + 1

    print(sat_list)
