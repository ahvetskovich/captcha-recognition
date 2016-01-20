import datetime
import logging
# myList = [0, 1, 2, 3, 4]
# i = 0
# while i < len(myList):
#     print (myList[i])
#     del myList[i]

date = datetime.datetime.now().strftime("%d-%m-%Y %H.%M")
logPath = 'E:/GitHub/captcha-recognition/ajax_captcha/log_partition_' + date + '.txt'
logging.basicConfig(filename=logPath, level=logging.DEBUG)
logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('And this, too')