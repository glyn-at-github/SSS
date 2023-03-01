import multiprocess

import time

start = time.time()

def count1000():

    i = 0
    while i < 500000000:
        i += 1
    print(i, "Finished!")
    return


# p1 = multiprocessing.Process(target=count1000)
# p2 = multiprocessing.Process(target=count1000)

# p1.start()
# p2.start()

# p1.join()
# p2.join()

count1000()

print("finished!")

end = time.time()

time_taken = end - start
print(time_taken)

quit()
