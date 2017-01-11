import os


class CsvToTxt:
    def __init__(self):
        return

    def csv2txt(inputF):
        with open(inputF, "r") as csvfile:
            csvf_ = csvfile.read().splitlines()
            path = inputF.split('/')
            path.pop()
            path.append('bothchroma.txt')
            outputF = '/'.join(path)

            a = csvf_[0].split(',')
            a.pop(0)
            csvf_[0] = ',' + ','.join(a)

            wtr = open(outputF, 'w')
            for r in csvf_:
                wtr.write(r[1:])
                wtr.write('\n')

        csvfile.close()
        wtr.close()
        return

    t2=[]
    rootDir = './McGill_Billboard'
    for dirpath, subdirList, fileList in os.walk(rootDir, topdown=True):
        for fname in fileList:
            if fname == 'bothchroma.csv':
                t2.append(os.path.join(dirpath, fname))
    t2.sort()
    print '#files bothchroma.csv: ', len(t2), '\n'

    for i in range(len(t2)):
        csv2txt(t2[i])

    print '#files bothchroma.txt: ', i+1, '\n'
    print ('done \n')