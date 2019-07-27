import json

json_string = {"last_name":u"Rossum", "first_name": u"Guido", }
ofile = open("./data.json",'w')
json.dump(json_string, ofile,  indent=4, sort_keys=False, ensure_ascii=True)
#ofile.close()

import io

ifile = io.open("./TestCFD.json",'r', encoding='utf8')
d = json.load(ifile)
print(d)