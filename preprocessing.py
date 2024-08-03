# -*- coding: utf-8 -*-
"""preprocessing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KSOgvG0gd0V8WV4o_8sff3a-51U4BWRv
"""

# Usage: python3 preprocessing.py "<path to >/preprocessed_dev/" "<path to jsonl>"

import os

fileindex = 0
base_path = sys.argv[1] # Originally just "./preprocessed_dev/"
filelimit = 1000


import json
with open(sys.argv[2]) as json_file:
    for line in json_file:
      data = (json.loads(line))
      title = data["title"]
      text = data["text"]
      summary = data["summary"]
      
      superfolder = os.path.join(base_path,str(int(int(fileindex)/int(filelimit))))
      subfolder = os.path.join(superfolder,str(fileindex % filelimit))

      if not os.path.exists(superfolder):
          os.mkdir(superfolder)
      if not os.path.exists(subfolder):
          os.mkdir(subfolder)
      
      textpath = os.path.join(subfolder,"text.txt")
      summarypath = os.path.join(subfolder,"summ.txt")
      
      textfile = open(textpath,"w+")
      textfile.write(title)
      textfile.write(text)
      textfile.close()
      
      summfile = open(summarypath,"w+")
      summfile.write(summary)
      summfile.close()
      fileindex += 1
      
      #Uncomment to limit number of files processed
      if fileindex == 10000:
          break
     
    json_file.close()