# AA em_l, em_h, se_l, se_h check 55
import serial
import time


def add_prefix(var):
    var = '0x' + var
    return var


# em是电机的,se是舵机的,通过电机的脉冲值和舵机的脉冲值算这个字符串.
def car_drive(em, se):
    em = int(em)
    se = int(se)
    if em > 2000:
        em = 2000
    elif em < 1000:
        em = 1000
    if se > 2000:
        se = 2000
    elif se < 1000:
        se = 1000
    em = hex(em).split('0x')[-1]
    se = hex(se).split('0x')[-1]
    if len(em) == 3:
        em_l = em[-2:].upper()
        em_h = '0' + em[-3:-2].upper()
    else:
        em_l = em[-2:].upper()
        em_h = em[-4:-2].upper()
    if len(se) == 3:
        se_l = se[-2:].upper()
        se_h = '0' + se[-3:-2].upper()
    else:
        se_l = se[-2:].upper()
        se_h = se[-4:-2].upper()
    check = int(add_prefix(em_l), 16) + int(add_prefix(em_h), 16) + int(add_prefix(se_l), 16) + int(add_prefix(se_h),
                                                                                                    16)
    check = hex(check).split('0x')[-1][-2:]
    result = 'AA ' + em_l + ' ' + em_h + ' ' + se_l + ' ' + se_h + ' ' + check + ' ' + '55'
    result = bytes.fromhex(result)
    return result
