with open('data_cleaner.py', 'r') as f:
    content = f.read()
with open('data_cleaner.py', 'w') as f:
    f.write(content.replace('                else:', '                    else:'))
