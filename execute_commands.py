#!/usr/bin/env python -W ignore::DeprecationWarning
'''
Execute a bag of commands.
Created on 07/02/2016
Updated on 13/04/2017: ignore commented lines
@author: reifortes
'''

import sys, os, time, random
from threading import Thread
from queue import Queue, Empty

# import util
queue = Queue()

import csv
import resource, gc
from datetime import datetime


def using(point=""):
    now = datetime.now()
    dtstr = now.strftime("%d/%m/%Y %H:%M:%S")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    print('\n***** %s\n%s: usertime = %f; systime = %f; mem= %.2f' % (
    dtstr, point, usage[0], usage[1], (usage[2] * resource.getpagesize()) / 1000000.0))
    gc.collect()


def run_command(command):
    f = os.system(command)
    return f


class ExecuteThread(Thread):
    def run(self):
        st = time.process_time()
        global queue
        while True:
            try:
                command = queue.get()
                if command == None:
                    queue.put(None)
                    break
            except Empty:
                continue
            print("- %s: %s" % (self.getName(), command))
            run_command(command)
            queue.task_done()
            if queue.empty(): time.sleep(random.random())
        et = time.process_time()
        ds = et - st
        dm = ds / 60
        print("\n%s Finished in %2.2f sec / %2.2f min (CPU time)" % (self.getName(), ds, dm))


if __name__ == '__main__':
    using("Inicio")
    st = time.time()
    print('Parameters: ' + str(sys.argv))
    commandsFile = open(sys.argv[1], "r")
    numberOfThreads = int(sys.argv[2])
    # reading commands
    produced = 0
    for line in commandsFile:
        line = line.strip()
        if not line or line.startswith('#'): continue
        queue.put(line)
        produced += 1
    commandsFile.close()
    queue.put(None)
    print("\nProduced %d commands.\n" % (produced))
    # creating threads
    consumers = []
    for i in range(0, min(numberOfThreads, produced)):
        consumer = ExecuteThread()
        consumers.append(consumer)
        consumer.start()
    # waiting threads
    for consumer in consumers:
        consumer.join()
    et = time.time()
    ds = et - st
    dm = ds / 60
    print("\nFinished in %2.2f sec / %2.2f min (EXE time)." % (ds, dm))
    using("Fim")
