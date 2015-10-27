import os
def Test1(rootDir):
    # print rootDir
    list_dirs = os.walk(rootDir)
    for root, dirs, files in list_dirs:
        # for d in dirs:
        #     print os.path.join(root, d)
        for f in files:
            if f[-4:]=='tiff':
                print os.path.join(root, f)

Test1('/Users/Xiangyu/Documents/bio/hw2/biometrics/assignment2/2008-03-11_13-2')
