# Loads contents of the config file

f = open("config", "r")
content = f.read()
f.close()
config = {}
for pair in content.split("\n"):
    if pair != "":
        items = pair.split(": ")
        config[items[0]] = items[1]
