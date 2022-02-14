import pdfplumber
from langdetect import detect
import os


pdfs = ["swift3.pdf"] #english
pdf = ["web4.pdf"] #russian
text= ""

'''
#read into local directory, ingest all PDF text
for file in os.listdir(os.getcwd()):
   with open(os.path.join(os.getcwd(), file), 'r') as f: # open in readonly mode
      filename = os.path.splitext(str(file))
      filetype = filename[1]
      if filetype == ".pdf":
      	with pdfplumber.open(file) as pdf:
      		for page in pdf.pages:
      			text += str(page.extract_text())

print(text)
'''

for i in pdfs:
	with pdfplumber.open(i) as pdf:
		for page in pdf.pages:
			text += str(page.extract_text())

print(text)

