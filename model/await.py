import pytz
from datetime import datetime, time
from time import sleep


tz = pytz.timezone('Europe/Amsterdam')

wait = True
while wait:
    ams_now = datetime.now(tz)
    if time(hour=15, minute=0) <= ams_now.time() <= time(hour=15, minute=17):
        print(f'Await, AMS time is now {ams_now}')
        sleep(15)
    else:
        wait = False
