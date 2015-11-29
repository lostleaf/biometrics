import csvio

listpath = "J:/Biometrics/images/image-metadata-2.0.csv"
pics = csvio.readList(listpath)

csvio.writeList("frontface_list.csv", pics)
