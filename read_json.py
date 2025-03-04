import json

rel = "logs\\json_dump.log"
logs_as_dictionary = {}

with open(rel,  'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):


        if line == "Sandbox logs:":
            continue
        if line == "Submission logs:":
            continue
        if not line:
            continue

        print(line)
        print(type(line))
        # split the line in time stamp and trading state
        [time_stamp, state] = line.split(' ',1) 
        print(time_stamp)

        if not state:
            logs_as_dictionary[time_stamp] = json.loads(state)
            print(time_stamp)

