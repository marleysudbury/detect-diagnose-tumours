# Loads contents of the config file

f = open("../config", "r")
content = f.read()
f.close()
config = {}
for pair in content.split("\n"):
    items = pair.split(": ")
    config[items[0]] = items[1]
