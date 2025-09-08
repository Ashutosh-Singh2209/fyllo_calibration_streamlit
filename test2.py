def find_median_index(lst):
    sorted_lst = sorted(lst)
    n = len(lst)
    
    if n % 2 == 1:
        median_value = sorted_lst[n // 2]
    else:
        median_value = (sorted_lst[n // 2 - 1] + sorted_lst[n // 2]) / 2
    
    min_diff = float('inf')
    median_index = 0
    
    for i, value in enumerate(lst):
        diff = abs(value - median_value)
        if diff < min_diff:
            min_diff = diff
            median_index = i
    
    return median_index

def find_calibration_points4(a, date_times):
    calibration_points = []
    cal_indices = []
    
    for i in range(20, len(a)):
        if a[i] - a[i-1] <= 0:
            last_non_positive_idx = None
            irrigation_idx = None
            
            for j in range(i, -1, -1):
                if a[j] - a[j-1] < -0.01:
                    last_non_positive_idx = j
                elif a[j] - a[j-1] > 0 :
                    break
            
            if last_non_positive_idx is not None:
                for j in range(last_non_positive_idx-1, -1, -1):
                    if a[j] - a[j-1] >= 0.01:
                        irrigation_idx = j
                        break
            
            if irrigation_idx is not None and last_non_positive_idx is not None and\
                  last_non_positive_idx <= i - 18:
                median_index = find_median_index(a[last_non_positive_idx+16:last_non_positive_idx+18])
                median_index = last_non_positive_idx+16+median_index
                calibration_points.append(date_times[median_index])
                cal_indices.append(median_index)
    
    if cal_indices:
        avg = sum(a[idx] for idx in cal_indices) / len(cal_indices)
    else:
        avg = None
    
    return calibration_points, avg



# def find_calibration_points4(a, date_times):
#     calibration_points = []
#     cal_indices = []
#     i = 1
    
#     while i < len(a):
#         if a[i] - a[i-1] >= 0.01 and a[i] - a[i-1] <= 0.1 :
#             while i < len(a) - 1 and a[i] - a[i-1] >= 0:
#                 i += 1
            
#             if i < len(a) and a[i] - a[i-1] < 0:
#                 t_0 = i
#                 peak_idx = None
                
#                 while i < len(a) - 1 and i - t_0 < 17:
#                     if a[i] - a[i-1] >= 0.01 and a[i] - a[i-1] <= 0.1\
#                           and peak_idx is not None:
#                         peak_idx = i
#                     i += 1
                
#                 if i - t_0 >= 17:
#                     median_index = (t_0 + i + 1) // 2
#                     calibration_points.append(date_times[median_index])
#                     cal_indices.append(median_index)
                
#                 if peak_idx is not None:
#                     i = peak_idx
#         else:
#             i += 1
    
#     if cal_indices:
#         sum_ = sum(a[idx] for idx in cal_indices)
#         avg = sum_ / len(cal_indices)
#     else:
#         avg = None
    
#     return calibration_points, avg

    # return calibration_points

# def find_calibration_points4(a, date_times, a_):
#     calibration_points = []
#     cal_indices = []
#     t_0 = None
#     i = 1

#     peak_idx = None
    
#     while i < len(a):
        
#         if a[i] > 0 and (a[i-1] <= 0):
#           while i < len(a) -1 and a[i] <= a[i+1] :
#               i+=1
#           if (a[i-1] <= a[i]):
#             t_0 = i
#             while i < len(a) -1 and i - t_0 < 17:
#                 i+=1
#             if  i - t_0 >= 17 :
#                 median_index = (i + (i-2))//2
#                 calibration_points.append(date_times[median_index])
#                 cal_indices.append(median_index)
        
        
#         i += 1
#     sum_ = 0
#     for idx in cal_indices:
#         sum_ += a_[idx] 
    
#     return calibration_points, sum_/len(cal_indices)


def find_calibration_points2(a, date_times):
    calibration_points = []
    irrigation_detected=False

    # for i in range(1, len(a), 1):
    i = 0
    while i < len(a):
        if a[i] > 0:
            irrigation_detected=True
        if ((a[i] > 0 and (a[i]>0.01) and irrigation_detected)) or ((i>0 and i<len(a)-1) and (a[i]>0.007) \
                                                                    and (a[i]>a[i-1]) and (a[i]>a[i+1])):
            valid_irr_point = i
            min_point = None
            while a[i] >= 0 and i < len(a) - 1:
                i+=1
            if a[i] <= 0:
                min_point=i
                while a[i] <= 0 and i < len(a) - 1:
                    # if a[i] < -0.003 :
                    #     break
                    if a[i] < a[min_point]:
                        min_point = i
                        if i<len(a)-1 and a[i] < a[i+1] and a[i] < -0.0009:
                            break
                    i+=1
                i = min_point + 1
                if a[min_point] < 0:
                    while i < len(a) - 1  and a[i] < -0.006 :
                        i += 1      
                    if i < len(a) and a[i] >= - 0.006 and a[i] <= 0:
                        calibration_points.append(date_times[i])
                        irrigation_detected=False
                    else:
                        i = valid_irr_point+1
                else:
                    i = min_point

        i += 1
    return calibration_points


def find_calibration_points3(a, date_times):
    calibration_points = []
    irrigation_detected=False

    # for i in range(1, len(a), 1):
    i = 0
    while i < len(a):
        if a[i] > 0:
            irrigation_detected=True
        if a[i] > 0 and (a[i]>0.01) and irrigation_detected:
            valid_irr_point = i
            min_point = None
            while a[i] >= 0 and i < len(a) - 1:
                i+=1
            if a[i] <= 0:
                min_point=i
                while a[i] <= 0 and i < len(a) - 1:
                    # if a[i] < -0.003 :
                    #     break
                    if a[i] < a[min_point]:
                        if not a[i] < 1.4*a[min_point]:
                            break
                        min_point = i
                    i+=1
                i = min_point + 1
                if a[min_point] < 0:
                    while i < len(a) - 1  and a[i] < -0.006 :
                        i += 1      
                    if i < len(a) and a[i] >= - 0.006 and a[i] <= 0:
                        calibration_points.append(date_times[i])
                        irrigation_detected=False
                    else:
                        i = valid_irr_point+1
                else:
                    i = min_point

        i += 1
    return calibration_points


if __name__ == '__main__':
    # a = [0, 0.016, 0, -0.0207, -0.0203, -0.00221]
    # indices = list(range(len(a)))
    # a = [-0.00222, 0.0108, -0.0043, 0, -0.00222]
    # indices = list(range(len(a)))
    # a = [0.01611, 0, -0.02067, -0.02034, -0.00221]
    # indices = list(range(len(a)))
    # a=[0.0235, 0, 0.01002, 0, 0, 0, -0.00262, -0.01161, -0.00787, -0.00783, -0.0021]
    # a=[0.01611, 0, -0.02067, -0.02034, -0.00221]
    # ans = -0.00221
    # a=[0.04311, -0.01148, 0.01611, 0, -0.02067, -0.02034, -0.00221]
    # ans = -0.00221
    # a=[0, 0, 0.04311, -0.01148, 0.01611, 0, -0.02067, -0.02034, -0.00221]
    # ans = -0.00221
    # a=[0.04141, 0.0111, 0, 0, 0.04311, -0.01148, 0.01611, 0, -0.02067, -0.02034, -0.00221]
    # ans = -0.00221
    # a=[0.00863, 0, 0, -0.00201, 0, 0, 0.00865, 0.04141, 0.0111, 0, 0, 0.04311, -0.01148, 0.01611, 0, -0.02067, -0.02034, -0.00221]
    # ans = -0.00221
    # a=[0.02519, 0, -0.00151, 0.04433, 0, 0, 0, 0, -0.00943, -0.00938, -0.01406, -0.00624, -0.00303]
    # ans = -0.00303
    a=[1.03386, 1.05132, 1.04989, 1.04747, 1.04328, 1.04072, ]
    ans = -0.00183
    indices = list(range(len(a)))
    cal_indices, ans_ = find_calibration_points4(a, indices)
    if cal_indices:
        print(a[cal_indices[0]] == ans)
        print(a[cal_indices[0]])