import csv


def filterList(filepath, glasses=False, profile=False):
    piclist = []
    with open(filepath, 'rU') as f:
        f_csv = csv.reader(f, dialect=csv.excel_tab)
        header = next(f_csv)
        pre = None
        for row in f_csv:
            row = row[0].split(",")
            picname = row[0].strip()
            pid = int(picname.split("d")[0])
            if pre!=pid:
                print(pid)
                pre = pid
            if pid == 90003:
                continue
            twin = 'A' if (pid%2)==0 else 'B'
            tid = int((pid - 90002) // 2)

            #print row
            
            withGlass = (row[6].strip()!="none")
            yaw = abs(int(row[8].strip()))
            
            #   Skip profile face
            if yaw == 90 or (not profile and yaw > 0):
                continue
            if withGlass and not glasses:
                continue
         
            piclist.append([picname, tid, twin])



    return piclist


def writeList(savepath, piclist):
    header = ['filename', 'id', 'twin']
    with open(savepath,'wb') as f:
        f_csv = csv.writer(f, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        f_csv.writerow(header)
        for line in piclist:
            #print line
            f_csv.writerow(line)


def readList(filepath):
    piclist = []
    with open(filepath, 'r') as f:
        f_csv = csv.reader(f, dialect=csv.excel_tab)
        header = next(f_csv)
        for row in f_csv:
            piclist.append(row)

    return piclist
