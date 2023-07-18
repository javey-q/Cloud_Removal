import  csv

def get_filelists_from_csv(listpath, phase):
    csv_file = open(listpath, "r")
    list_reader = csv.reader(csv_file)

    train_filelist = []
    val_filelist = []
    test_filelist = []
    for f in list_reader:
        line_entries = f
        if line_entries[0] == '1':
            train_filelist.append(line_entries)
        elif line_entries[0] == '2':
            val_filelist.append(line_entries)
        elif line_entries[0] == '3':
            test_filelist.append(line_entries)
    csv_file.close()
    # to do
    if phase == 'train':
        return train_filelist
    elif phase == 'val':
        return val_filelist
    elif phase == 'test':
        return test_filelist