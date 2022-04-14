import linecache
import os
import shutil
import random

def split_dataset(path, copies):
    train_filepath = path + '/train.txt'
    test_filepath = path + '/test.txt'
    with open(train_filepath, "r") as f:
        with open(test_filepath, 'r') as f1:
            lines = len(f.readlines())
            offset = int(lines / copies)
            print('[*] The oriented file contain lines: %d' % lines),
            print('[*] The new file will contain lines: %d' % offset)

            newfolder_path = path + '/split_set'
            if not os.path.exists(newfolder_path):
                os.makedirs(newfolder_path)
                print('[*] create a new folder!')
            else:
                shutil.rmtree(newfolder_path)  # Removes all the subdirectories!
                os.makedirs(newfolder_path)

            for i in range(1, copies+1):
                resultList = random.sample(range(0, lines), offset)
                with open(newfolder_path+"/dataset_train-{}.txt".format(i), 'w') as f2:
                    for j in resultList:
                        f2.write(linecache.getline(train_filepath, j))
                    print('[*] Successfully created file dataset_train-{}!'.format(i))
                with open(newfolder_path + "/dataset_test-{}.txt".format(i), 'w') as f2:
                    for j in resultList:
                        f2.write(linecache.getline(test_filepath, j))
                    print('[*] Successfully created file dataset_test-{}!'.format(i))

    f.close()
    f1.close()
    f2.close()
    print('[*] Data split finish!')

if __name__ == '__main__':
    split_dataset('Data/ml-1m', 10)